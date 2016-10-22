package BayesianIBM;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import sampling.MultivariateLikelihood;
import sampling.MultivariateSliceSampler;
import sampling.ProductOfSymmetricDirichletLikelihoods;
import alignmentUtils.ParallelCorpus;
import collections.DoubleCounter;
import collections.IntCountTable;
import collections.IntCounter;
import collections.machineTranslation.LogTranslationTable;
import ibmModels.IBM1;
import io.TranslationTableUtils;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

public class CollocationIBM2 extends GibbsIBM2 {

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

	private MultivariateSliceSampler<Integer> collocationHyperSampler;

	public CollocationIBM2(double translationPrior, double distortionPrior, double lmPrior) {
		super(distortionPrior);
		this.collocationIndicators = new ArrayList<boolean[]>();
		this.collocationTable = new IntCountTable();
		this.continuations = new IntCounter();
		this.sourceWordFreqs = new IntCounter();
		setTranslationPrior(translationPrior);
		setLMPrior(lmPrior);
		this.collocationHyperSampler = new MultivariateSliceSampler<Integer>(new GammaDistribution(1, 10), null);
	}

	@Override
	public void setDistortionPrior(double distortionPrior) throws IllegalArgumentException {
		if (distortionPrior <= 0) {
			throw new IllegalArgumentException("The distortionPrior (Dirichlet parameter) must be positive.");
		}
		this.distortionPrior = new Int2DoubleOpenHashMap();
		this.distortionPrior.put(0, distortionPrior);

		for (int dist = -maxDistortion; dist <= maxDistortion; dist++) {
			this.distortionPrior.put(dist, distortionPrior);
		}
	}

	public void setCollocationPrior(double a, double b) {
		this.collocationPrior = b;
		this.betaTotal = a + b;
	}

	public void setLMPrior(double prior) {
		this.lmPrior = prior;
	}

	private int newRandomLink(int frenchPos, int frenchLength, int englishLength) {
		double[] alignmentProbs = new double[englishLength];
		double total = 0;
		for (int pos = 1; pos < englishLength; pos++) {
			double score = scoreDistortion(frenchPos, pos, frenchLength, englishLength);
			total += score;
			alignmentProbs[pos] = score;
		}

		int newLink = -1;
		for (int pos = 1; pos < englishLength; pos++) {
			newLink = pos;
			if ((total - alignmentProbs[pos]) <= 0) {
				break;
			}
		}
		return newLink;
	}

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
			int frenchLength = frenchSide.length;
			int[] currentAlignment = new int[frenchLength];
			boolean[] collocationVector = new boolean[frenchLength];

			// -2 discounts the NULL position which has a separate distortion value
			this.maxDistortion = englishSide.length - 2 > this.maxDistortion ? englishSide.length - 2 : this.maxDistortion;

			int prevWord = SOURCE_START_SYMBOL;
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

					// randomly choose new link according to previously seen links
					int randomLink = newRandomLink(j, frenchSide.length, englishSide.length);
					currentAlignment[j] = randomLink;
					this.distortionCounter.put(this.computeDistortion(j, randomLink, frenchSide.length, englishSide.length), 1.0);
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
		setDistortionPrior(this.distortionPrior.get(0));

		// initialize indicators
		System.out.println("Starting to initialise collocation indicators at " + new Date());
		for (int iter = 1; iter <= 200; iter++) {
			sampleCollocationIndicatorsOnce(corpus);
		}
		System.out.println("Finished initialising collocation indicators at " + new Date());
	}

	protected void removeCollocationInformation(int prevWord, int currentWord) {
		this.continuations.subtract(prevWord, 1.0);
		this.collocationTable.subtract(prevWord, currentWord, 1.0);
	}

	protected void updateCollocationInformation(int prevWord, int currentWord) {
		this.continuations.put(prevWord, 1.0);
		this.collocationTable.put(prevWord, currentWord, 1.0);
	}

	protected void sampleCollocationIndicatorsOnce(ParallelCorpus corpus) {
		for (int sent = 0; sent < corpus.size(); sent++) {
			int[][] pair = corpus.get(sent);
			int[] frenchSide = pair[0];
			int[] englishSide = pair[1];

			boolean[] collocationVector = this.collocationIndicators.get(sent);
			int[] alignmentVector = this.state.get(sent);

			int prevWord = this.SOURCE_START_SYMBOL;
			boolean wasCollocated;

			for (int j = 0; j < frenchSide.length; j++) {
				int fword = frenchSide[j];
				int englishPos = alignmentVector[j];

				if (wasCollocated = collocationVector[j]) {
					removeCollocationInformation(prevWord, fword);
				} else {
					this.removeInformation(englishPos, englishSide, j, frenchSide);
				}

				boolean newIndicator = sampleCollocationIndicator(englishPos, englishSide, j - 1, j, frenchSide);

				if (newIndicator) {
					updateCollocationInformation(prevWord, fword);
					if (!wasCollocated) {
						this.distortionCounter.put(computeDistortion(j, englishPos, frenchSide.length, englishSide.length), 1.0);
					}
				} else if (wasCollocated) {
					// indicator has changed from collocation to alignment -> thus no jump needs to be added
					int englishWord = englishSide[englishPos];

					int sourceCount = (int) Math.round(this.translationCountTable.get(englishWord, frenchSide[j]));
					if (this.conditionalSourceCounts.containsKey(sourceCount)) {
						this.conditionalSourceCounts.subtract(sourceCount, 1.0);
					}
					this.conditionalSourceCounts.put(sourceCount + 1, 1.0);

					this.translationCountTable.put(englishWord, frenchSide[j], 1.0);
					this.englishTotals.put(englishWord, 1.0);
				} else {
					this.updateInformation(alignmentVector[j], englishSide, j, frenchSide, alignmentVector);
				}
				collocationVector[j] = newIndicator;
				prevWord = fword;
			}
		}
	}

	private boolean sampleCollocationIndicator(int englishPos, int[] englishSide, int prevFrenchPos, int currentFrenchPos, int[] frenchSide) {
		int prevWord = prevFrenchPos == -1 ? this.SOURCE_START_SYMBOL : frenchSide[prevFrenchPos];
		int currentWord = frenchSide[currentFrenchPos];
		int alignedWord = englishSide[englishPos];

		double continuationProb = this.continuations.get(prevWord) + this.collocationPrior;
		double collocationProb = (this.collocationTable.get(prevWord, currentWord) + this.lmPrior)
				/ (this.continuations.get(prevWord) + this.lmPriorTotal);
		// double distortionProb = this.scoreDistortion(currentFrenchPos, englishPos, frenchSide.length, englishSide.length);
		double alignProb = (this.translationCountTable.get(alignedWord, currentWord) + this.childConcentration)
				/ (this.englishTotals.get(alignedWord) + this.translationPriorTotal);
		double alignmentProb = alignProb;

		double threshold = this.randomGenerator.nextDouble() * (sourceWordFreqs.get(prevWord) + betaTotal)
				* (alignmentProb + collocationProb);
		return continuationProb * collocationProb > threshold;
	}

	protected void sample(ParallelCorpus corpus, int iterations, int burnIn, int lag) throws IOException {
		double decile = Math.floor(iterations / 10.0);
		HashMap<Integer, int[]> positionsPerSentenceLength = new HashMap<Integer, int[]>();

		this.sizeOfSupport = sourceWordFreqs.size();
		int samplesTaken = 0;
		// an upper bound on the number of samples to be taken in total

		this.translationPriorTotal = baseDistribution == null ? childConcentration * sizeOfSupport : childConcentration;

		// stores the probabilities for no co-occurrences with a given total count
		DoubleCounter probabilityCache = new DoubleCounter();
		probabilityCache.put(0.0, childConcentration / translationPriorTotal);

		System.out.println("Started sampling at " + new Date());

		for (int iter = 1; iter <= iterations; iter++) {
			if (iter % decile == 0) {
				System.out.println("Iteration " + iter);
			}

			long start = System.nanoTime();
			sampleAlignmentsOnce(corpus, positionsPerSentenceLength, probabilityCache, translationPriorTotal);
			sampleCollocationIndicatorsOnce(corpus);
			System.out.printf("Gibbs iteration took %f seconds%n", (System.nanoTime() - start) / 1000000000.0);

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
				System.out.printf("Log-Likelihood at iteration %d: %f%n", iter, this.computeLogLikelihood());
			}
		}
		System.out.println("Finished sampling at " + new Date());
		System.out.printf("%d samples taken in total.%n", samplesTaken);
	}

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
					this.distortionCounter.subtract(computeDistortion(j, alignmentVector[j], frenchSide.length, englishSide.length), 1.0);
					double[] probs = new double[competition.length];
					double total = 0;
					for (int i = 0; i < competition.length; i++) {
						double score = this.scoreDistortion(j, competition[i], frenchSide.length, englishSide.length);
						total += score;
						probs[i] = score;
					}

					double threshold = total * this.randomGenerator.nextDouble();
					for (int i = 0; i < competition.length; i++) {
						if ((threshold -= probs[i]) <= 0) {
							newLink = competition[i];
							break;
						}
					}
					alignmentVector[j] = newLink;
					this.distortionCounter.put(computeDistortion(j, newLink, frenchSide.length, englishSide.length), 1.0);
				} else {
					int currentLink = alignmentVector[j];
					this.removeInformation(currentLink, englishSide, j, frenchSide);

					// TODO check whether IBM2 gets called here
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

					if (newLink == -1) {
						for (int item : competition) {
							System.out.println("Target = " + englishSide[item]);
							for (Map.Entry<Integer, Double> entry : this.translationCountTable.getRow(englishSide[item]).entrySet()) {
								System.out.println(entry.getKey() + " = " + entry.getValue());
							}
						}
					}
					this.updateInformation(newLink, englishSide, j, frenchSide, alignmentVector);
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
	
	@Override
	// make sure that alignments to NULL never have positive probability -- needed when aux is not used
	public double scoreDistortion(int sourcePos, int targetPos, int sourceLength, int targetLength) {
		return targetPos == 0 ? 0 : super.scoreDistortion(sourcePos, targetPos, sourceLength, targetLength);
	}
	
	// ////////////////////// TODO This is only here as a quick fix and needs to be adjusted later

	protected double[][] getAlignmentProbs(int[] englishSide, int[] competition, int[] frenchSide, int frenchPos,
			DoubleCounter probabilityCache, double priorTotal) {
		double probs[] = new double[englishSide.length];
		double totalProbs = 0;
		int frenchWord = frenchSide[frenchPos];

		for (int i : competition) {
			int englishWord = englishSide[i];
			double translationCount = this.getTranslationCountTable().get(englishWord, frenchWord);
			double translationTotal = this.getEnglishTotals().get(englishWord);

			double alignmentProb = this.scoreTranslationPair(englishWord, frenchWord, translationCount, translationTotal, priorTotal,
					probabilityCache);

			alignmentProb *= scoreDistortion(frenchPos, i, frenchSide.length, englishSide.length);
			probs[i] = alignmentProb;
			totalProbs += alignmentProb;
		}
		return new double[][] { probs, new double[] { totalProbs } };
	}

	// //////////////////////////////////////////////////////////////////////////////////////////////

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
	public double computeLogLikelihood() {
		double likelihood = super.computeLogLikelihood();
		return likelihood;
	}

	@Override
	protected void sliceSampleHyperparameters(int iterations) {
		super.sliceSampleHyperparameters(iterations);
		sampleLMHyperparameters(iterations);
		sampleCollocationHyperparameters(iterations);
	}

	private void sampleLMHyperparameters(int iterations) {
		IntCounter lmConditionalCounts = computeLMConditionalCounts();

		this.translationHyperSampler.setLikelihood(new ProductOfSymmetricDirichletLikelihoods(this.continuations.size(),
				this.continuations, lmConditionalCounts, this.sizeOfSupport));
		this.translationHyperSampler.sample(this.lmPriorTotal, iterations, 0, 0);
		double newValue = this.translationHyperSampler.getState();

		this.lmPriorTotal = newValue;
		this.lmPrior = newValue / this.sizeOfSupport;

		System.out.println("lmPrior = " + this.lmPrior);
	}

	private void sampleCollocationHyperparameters(int iterations) {
		this.collocationHyperSampler.setLikelihood(new BinomialLikelihood(this.continuations, this.sourceWordFreqs));

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
