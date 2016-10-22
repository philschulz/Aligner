package BayesianIBM;

import ibmModels.IBM1;
import io.TranslationTableUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import sampling.HyperparameterSliceSampler;
import sampling.MultivariateLikelihood;
import sampling.MultivariateSliceSampler;
import sampling.ProductOfSymmetricDirichletLikelihoods;
import collections.DoubleCounter;
import collections.IntCountTable;
import collections.IntCounter;
import collections.machineTranslation.LogTranslationTable;
import alignmentUtils.ParallelCorpus;

public class CollocationIBM1 extends GibbsIBM1 {

	private final int SOURCE_START_SYMBOL = -1;
	private List<boolean[]> collocationIndicators;
	private double collocationPrior;
	private double betaTotal;
	private double lmPrior;
	private double lmPriorTotal;
	// stores for each word pair the number of times the second word has followed the
	// first in a collocation
	private IntCountTable collocationTable;
	// stores for each word the number of times it has been extended to a collocation
	private IntCounter continuations;
	// frequencies of all source words
	private IntCounter sourceWordFreqs;

	private HyperparameterSliceSampler LMHyperSampler;
	private MultivariateSliceSampler<Integer> collocationHyperSampler;

	public CollocationIBM1() {
		super();
		this.collocationIndicators = new ArrayList<boolean[]>();
		this.collocationTable = new IntCountTable();
		this.continuations = new IntCounter();
		this.sourceWordFreqs = new IntCounter();
	}

	public void setCollocationPrior(double a, double b) {
		this.collocationPrior = b;
		this.betaTotal = a + b;
	}

	public void setLMPrior(double prior) {
		this.lmPrior = prior;
	}

	// TODO adjust english side for NULL words
	@Override
	protected void assignInitialAlignment(ParallelCorpus corpus) throws IOException {

		IBM1 baseModel;
		if (!this.ibm1Table.equals("")) {
			LogTranslationTable<Integer> translationTable = TranslationTableUtils.readNumericTranslationTable(this.ibm1Table);
			baseModel = IBM1.initializeWith(translationTable);
		} else {
			baseModel = IBM1.createModel(corpus, 5);
			baseModel.writeTranslationTableToFile("translation_table_ibm1");
		}
		LogTranslationTable<Integer> tTable = baseModel.getLogTranslationTable();

		for (int[][] sentencePair : corpus) {
			int[] frenchSide = sentencePair[0];
			int[] englishSide = sentencePair[1];
			int[] currentAlignment = new int[frenchSide.length];
			boolean[] collocationVector = new boolean[frenchSide.length];
			int prevWord = -1;
			this.sourceWordFreqs.put(prevWord, 1.0);

			for (int j = 0; j < frenchSide.length; j++) {
				int frenchWord = frenchSide[j];
				this.sourceWordFreqs.put(frenchWord, 1.0);

				int alignmentPoint = -1;
				double highestScore = Double.MIN_EXPONENT;
				for (int i = 0; i < englishSide.length; i++) {
					double score = tTable.scoreTranslationPair(frenchWord, englishSide[i]);
					if (score > highestScore) {
						highestScore = score;
						alignmentPoint = i;
					}
				}
				if (alignmentPoint < 0) {
					System.out.println("Severe Error: There was no alignment point of French word " + frenchWord + ". "
							+ "This indicates that something is going wrong with IBM1.");
				} else if (alignmentPoint == 0) {
					collocationVector[j] = true;
					updateCollocationInformation(prevWord, frenchWord);
					currentAlignment[j] = this.randomGenerator.nextInt(englishSide.length - 1) + 1;
				} else {
					updateInformation(alignmentPoint, englishSide, j, frenchSide, currentAlignment);
				}
				prevWord = frenchWord;
			}
			this.state.add(currentAlignment);
			this.collocationIndicators.add(collocationVector);
		}

		if (this.assymetricIBM1Prior) {
			setBaseDistribution(tTable.toRealSpace());
		}

		this.lmPriorTotal = this.sourceWordFreqs.size() * this.lmPrior;

		// initialize indicators
		System.out.println("Starting to initialise collocation indicators at " + new Date());
		for (int iter = 1; iter <= 200; iter++) {
			sampleCollocationIndicatorsOnce(corpus);
		}
		System.out.println("Finished initialising collocation indicators at " + new Date());
	}

	@Override
	protected void takeSample() {
		if (this.samples.isEmpty()) {
			for (int[] sent : this.state) {
				List<IntCounter> sentence = new ArrayList<IntCounter>();
				for (@SuppressWarnings("unused")
				int link : sent) {
					sentence.add(new IntCounter());
				}
				this.samples.add(sentence);
			}
		}

		for (int sent = 0; sent < this.state.size(); sent++) {
			int[] alignmentVector = this.state.get(sent);
			boolean[] collocationVector = this.collocationIndicators.get(sent);
			List<IntCounter> sentenceSamples = this.samples.get(sent);
			for (int link = 0; link < alignmentVector.length; link++) {
				if (collocationVector[link]) {
					sentenceSamples.get(link).put(0, 1.0);
				} else {
					sentenceSamples.get(link).put(alignmentVector[link], 1.0);
				}
			}
		}
	}

	@Override
	protected int[] defineCompetition(int[] englishSide, int currentAlignedPos, int[] englishPositions) {
		int[] competition;

		if (this.aux) {
			// this is a hack to work with sentences only containing one word
			if (englishSide.length == 2) {
				return new int[] { currentAlignedPos, currentAlignedPos };
			}
			// this excludes sampling the NULL word!
			int competitor = randomGenerator.nextInt(englishSide.length - 1) + 1;
			if (competitor == currentAlignedPos) {
				return defineCompetition(englishSide, currentAlignedPos, englishPositions);
			} else {
				competition = new int[] { currentAlignedPos, competitor };
			}
		} else {
			competition = englishPositions;
		}

		return competition;
	}

	protected void sample(ParallelCorpus corpus, int iterations, int burnIn, int lag) throws IOException {
		double decile = Math.floor(iterations / 10.0);
		HashMap<Integer, int[]> positionsPerSentenceLength = new HashMap<Integer, int[]>();

		// correct for the fact that there is an initial token for each sentence that is never produced
		this.sizeOfSupport = sourceWordFreqs.size() - corpus.size();
		int samplesTaken = 0;
		this.translationPriorTotal = baseDistribution == null ? childConcentration * sizeOfSupport : childConcentration;

		// stores the probabilities for no co-occurrences with a given total count
		DoubleCounter probabilityCache = new DoubleCounter();
		probabilityCache.put(0.0, childConcentration / translationPriorTotal);

		System.out.println("Started sampling at " + new Date());

		for (int iter = 1; iter <= iterations; iter++) {
			if (iter % decile == 0) {
				System.out.println("Iteration " + iter);
			}

			long gibbStart = System.nanoTime();
			sampleAlignmentsOnce(corpus, positionsPerSentenceLength, probabilityCache, translationPriorTotal);
			sampleCollocationIndicatorsOnce(corpus);
			System.out.printf("Gibbs iteration took %f seconds%n", (System.nanoTime() - gibbStart) / 1000000000.0);

			// take a sample
			if (iter >= burnIn && iter % lag == 0 && !this.importanceSampler) {
				this.takeSample();
				samplesTaken++;
			}

			if (this.experimentPrintOut > 0 && iter % this.experimentPrintOut == 0) {
				this.printAlignmentsDuringExperiment(iter);
			}

			if (this.hyper) {
				long hyperStart = System.nanoTime();
				sliceSampleHyperparameters(10);
				System.out.printf("Slice sampling took %f seconds%n", (System.nanoTime() - hyperStart) / 1000000000.0);
				probabilityCache.clear();
			}

			if (this.printLikelihood) {
				System.out.printf("Log-Likelihood at iteration %d: %f%n", iter, computeLogLikelihood());
			}
		}

		System.out.println("Finished sampling at " + new Date());
		System.out.printf("%d samples taken in total.%n", samplesTaken);
	}

	/**
	 * Sample alignment variables conditional on the collocation variables. If a collocation is set to true, the alignment link is sampled
	 * according to the alignment distribution. Otherwise, standard Gibbs sampling for alignments is performed.
	 * 
	 * @param corpus
	 *            A ParallelCorpus
	 * @param positionsPerSentenceLength
	 *            A map from target sentence Lengths to arrays containing their positions
	 */
	protected void sampleAlignmentsOnce(ParallelCorpus corpus, Map<Integer, int[]> positionsPerSentenceLength,
			DoubleCounter probabilityCache, double translationPriorTotal) {
		for (int sent = 0; sent < corpus.size(); sent++) {
			int[][] pair = corpus.get(sent);
			int[] frenchSide = pair[0];
			int[] englishSide = pair[1];
			int[] englishPositions;

			if ((englishPositions = positionsPerSentenceLength.get(englishSide.length)) == null) {
				englishPositions = new int[englishSide.length];
				for (int i = 0; i < englishSide.length; i++) {
					englishPositions[i] = i;
				}
				positionsPerSentenceLength.put(englishSide.length, englishPositions);
			}

			boolean[] collocationVector = this.collocationIndicators.get(sent);
			int[] alignmentVector = this.state.get(sent);

			for (int j = 0; j < frenchSide.length; j++) {
				int[] competition = this.defineCompetition(englishSide, alignmentVector[j], englishPositions);
				int newLink = -1;

				if (collocationVector[j]) {
					newLink = competition[randomGenerator.nextInt(competition.length)];
					alignmentVector[j] = newLink;
				} else {
					int currentLink = alignmentVector[j];
					this.removeInformation(currentLink, englishSide, j, frenchSide);

					double[][] alignProbs = getAlignmentProbs(englishSide, competition, frenchSide, j, probabilityCache,
							translationPriorTotal);
					double[] probs = alignProbs[0];
					double totalProbs = alignProbs[1][0];

					double threshold = randomGenerator.nextDouble() * totalProbs;
					for (int item : competition) {
						if ((threshold -= probs[item]) <= 0) {
							newLink = item;
							break;
						}
					}
					this.updateInformation(newLink, englishSide, j, frenchSide, alignmentVector);
				}
			}
		}
	}

	@Override
	// make sure that NULL alignments never have positive probability -- needed when aux is not used 
	public double scoreTranslationPair(int englishWord, int frenchWord, double count, double total, double translationPriorTotal,
			DoubleCounter probabilityCache) {
		return englishWord == 0 ? 0 : super.scoreTranslationPair(englishWord, frenchWord, count, total, translationPriorTotal, probabilityCache);
	}
	
	protected void sampleCollocationIndicatorsOnce(ParallelCorpus corpus) {
		for (int sent = 0; sent < corpus.size(); sent++) {
			int[][] pair = corpus.get(sent);
			int[] frenchSide = pair[0];
			int[] englishSide = pair[1];

			boolean[] collocationVector = this.collocationIndicators.get(sent);
			int[] alignmentVector = this.state.get(sent);

			int prevWord = this.SOURCE_START_SYMBOL;

			for (int j = 0; j < frenchSide.length; j++) {
				int fword = frenchSide[j];

				if (collocationVector[j]) {
					removeCollocationInformation(prevWord, fword);
				} else {
					this.removeInformation(alignmentVector[j], englishSide, j, frenchSide);
				}

				boolean newIndicator = sampleCollocationIndicator(prevWord, fword, alignmentVector[j]);

				if (newIndicator) {
					updateCollocationInformation(prevWord, fword);
				} else {
					this.updateInformation(alignmentVector[j], englishSide, j, frenchSide, alignmentVector);
				}
				collocationVector[j] = newIndicator;
				prevWord = fword;
			}
		}
	}

	protected void removeCollocationInformation(int prevWord, int currentWord) {
		this.continuations.subtract(prevWord, 1.0);
		this.collocationTable.subtract(prevWord, currentWord, 1.0);
	}

	protected void updateCollocationInformation(int prevWord, int currentWord) {
		this.continuations.put(prevWord, 1.0);
		this.collocationTable.put(prevWord, currentWord, 1.0);
	}

	private boolean sampleCollocationIndicator(int prevWord, int currentWord, int alignedWord) {
		double continuationProb = this.continuations.get(prevWord) + this.collocationPrior;
		double collocationProb = (this.collocationTable.get(prevWord, currentWord) + this.lmPrior)
				/ (this.continuations.get(prevWord) + this.lmPriorTotal);
		double alignmentProb = (this.translationCountTable.get(alignedWord, currentWord) + this.childConcentration)
				/ (this.englishTotals.get(alignedWord) + this.translationPriorTotal);

		double threshold = this.randomGenerator.nextDouble() * (sourceWordFreqs.get(prevWord) + betaTotal)
				* (alignmentProb + collocationProb);
		return continuationProb * collocationProb > threshold;
	}

	// //////////////////////////////////////////// Hyperparameter Inference ///////////////////////////////////

	@Override
	public double computeLogLikelihood() {
		double likelihood = super.computeLogLikelihood();
		ProductOfSymmetricDirichletLikelihoods lm = new ProductOfSymmetricDirichletLikelihoods(this.continuations.size(),
				this.continuations, computeLMConditionalCounts(), this.sizeOfSupport);
		BinomialLikelihood colloc = new BinomialLikelihood(this.continuations, this.sourceWordFreqs);
		HashMap<Integer, Double> betaParams = new HashMap<Integer, Double>();
		betaParams.put(0, this.betaTotal - this.collocationPrior);
		betaParams.put(1, this.collocationPrior);

		if (this.hyper) {
			likelihood += lm.compute(this.lmPriorTotal) + this.LMHyperSampler.priorLogDensity(lmPriorTotal);
			likelihood += colloc.compute(betaParams) + this.collocationHyperSampler.priorLogDensity(betaParams);
		} else {
			likelihood += lm.compute(this.lmPriorTotal);
			likelihood += colloc.compute(betaParams);
		}
		return likelihood;
	}

	protected void sliceSampleHyperparameters(int iterations) {
		super.sliceSampleHyperparameters(iterations);
		sampleLMHyperparameters(iterations);
		sampleCollocationHyperparameters(iterations);
	}

	private void sampleLMHyperparameters(int iterations) {
		IntCounter lmConditionalCounts = computeLMConditionalCounts();

		if (this.LMHyperSampler == null) {
			this.LMHyperSampler = new HyperparameterSliceSampler(new GammaDistribution(this.translationGammaA, 1 / this.translationGammaB),
					new ProductOfSymmetricDirichletLikelihoods(this.continuations.size(), this.continuations, lmConditionalCounts,
							this.sizeOfSupport));
		} else {
			this.LMHyperSampler.setLikelihood(new ProductOfSymmetricDirichletLikelihoods(this.continuations.size(), this.continuations,
					lmConditionalCounts, this.sizeOfSupport));
		}

		this.LMHyperSampler.sample(this.lmPriorTotal, iterations, 0, 0);
		double newValue = this.LMHyperSampler.getState();

		this.lmPriorTotal = newValue;
		this.lmPrior = newValue / this.sizeOfSupport;

		System.out.println("lmPrior = " + this.lmPrior);
	}

	private void sampleCollocationHyperparameters(int iterations) {
		if (this.collocationHyperSampler == null) {
			this.collocationHyperSampler = new MultivariateSliceSampler<Integer>(new GammaDistribution(1, 1), new BinomialLikelihood(
					this.continuations, this.sourceWordFreqs));
		} else {
			this.collocationHyperSampler.setLikelihood(new BinomialLikelihood(this.continuations, this.sourceWordFreqs));
		}

		HashMap<Integer, Double> params = new HashMap<Integer, Double>();
		params.put(0, this.betaTotal - this.collocationPrior);
		params.put(1, this.collocationPrior);
		this.collocationHyperSampler.sample(params, iterations, 0, 0);
		Map<Integer, Double> newValue = this.collocationHyperSampler.getState();
		this.collocationPrior = newValue.get(1);
		this.betaTotal = this.collocationPrior + newValue.get(0);

		System.out.printf("Collocation alpha = %f, beta = %f%n", newValue.get(0), newValue.get(1));
	}

	private IntCounter computeLMConditionalCounts() {
		IntCounter conditionalCounts = new IntCounter();
		for (IntCounter condition : this.collocationTable.values()) {
			for (double value : condition.values()) {
				conditionalCounts.put((int) value, 1.0);
			}
		}
		return conditionalCounts;
	}

	private class BinomialLikelihood implements MultivariateLikelihood<Integer> {
		// TODO IMPORANT: beta is continuation hyperprior! Thus beta is prior on successes

		private IntCounter successes = new IntCounter();
		private IntCounter totals = new IntCounter();
		private double alpha;
		private double beta;

		public BinomialLikelihood(IntCounter successes, IntCounter totals) {
			this.successes = successes;
			this.totals = totals;
		}

		@Override
		public double compute(Map<Integer, Double> parameter) throws IllegalArgumentException {
			if (!(2 == parameter.size())) {
				throw new IllegalArgumentException("Length of parameter vector has to be exactly 2!");
			}
			this.alpha = parameter.get(0);
			this.beta = parameter.get(1);

			double result = this.totals.size() * (Gamma.logGamma(alpha + beta) - Gamma.logGamma(alpha) - Gamma.logGamma(beta));
			for (Map.Entry<Integer, Double> entry : this.totals.entrySet()) {
				double successes = this.successes.get(entry.getKey());
				double total = this.totals.get(entry.getKey());
				result += Gamma.logGamma(beta + successes) + Gamma.logGamma(alpha + total - successes)
						- Gamma.logGamma(alpha + beta + total);
			}
			return result;
		}

		@Override
		public double computeAt(Integer dimension, double value) {
			if (dimension < 0 || dimension > 2) {
				// TODO throw Exception
			}

			double alpha;
			double beta;

			if (dimension == 0) {
				alpha = value;
				beta = this.beta;
			} else {
				alpha = this.alpha;
				beta = value;
			}

			double result = this.totals.size() * (Gamma.logGamma(alpha + beta) - Gamma.logGamma(alpha) - Gamma.logGamma(beta));
			for (Map.Entry<Integer, Double> entry : this.totals.entrySet()) {
				double successes = this.successes.get(entry.getKey());
				double total = this.totals.get(entry.getKey());
				result += Gamma.logGamma(beta + successes) + Gamma.logGamma(alpha + total - successes)
						- Gamma.logGamma(alpha + beta + total);
			}
			return result;
		}

		@Override
		public void updateAt(Integer dimension, double value) {
			if (dimension < 0 || dimension > 2) {
				// TODO throw Exception
			}

			if (dimension == 0) {
				this.alpha = value;
			} else {
				this.beta = value;
			}
		}

		@Override
		public boolean hasObservation(Integer dimension) {
			return true;
		}
	}
}
