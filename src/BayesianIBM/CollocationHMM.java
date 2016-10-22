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

import sampling.MultivariateLikelihood;
import sampling.MultivariateSliceSampler;
import alignmentUtils.ParallelCorpus;
import collections.DoubleCounter;
import collections.IntCountTable;
import collections.IntCounter;
import collections.machineTranslation.LogTranslationTable;

public class CollocationHMM extends BayesHMM {

	protected final int SOURCE_START_SYMBOL = -1;
	protected List<boolean[]> collocationIndicators;
	protected double collocationPrior;
	protected double betaTotal;
	private double lmPrior;
	private double lmPriorTotal;
	protected IntCounter sourceWordFreqs;
	protected IntCountTable collocationTable;
	protected IntCounter continuations;
	private MultivariateSliceSampler<Integer> collocationHyperSampler;

	private boolean alignAll;
	private IntCounter sourceContextTotals;
	protected IntCounter continuationCounts;
	private boolean hyperAfterSample;

	public CollocationHMM(double translationPrior, double transitionPrior, double lmPrior) {
		super(translationPrior, transitionPrior);
		setLMPrior(lmPrior);
		this.sourceWordFreqs = new IntCounter();
		this.continuations = new IntCounter();
		this.collocationTable = new IntCountTable();
		this.collocationIndicators = new ArrayList<boolean[]>();
		this.alignAll = false;
		this.sourceContextTotals = new IntCounter();
		this.continuationCounts = new IntCounter();
		this.hyperAfterSample = false;
	}

	public void hyperAfterSample(boolean value) {
		this.hyperAfterSample = value;
		if (value) {
			this.hyper = false;
		}
	}
	
	public void setAlignAll(boolean alignAll) {
		this.alignAll = alignAll;
	}

	@Override
	public void setTransitionPrior(double prior) {
		IntCounter transitionPrior = new IntCounter();
		for (int i = -maxJump; i <= maxJump; i++) {
			transitionPrior.put(i, prior);
		}
		this.transitionPrior = transitionPrior;
	}

	@Override
	protected int computeJump(int prevPos, int currentPos) {
		if (prevPos == this.START_STATE) {
			return currentPos;
		} else {
			// forward jumps are positive, backward jumps are negative
			return currentPos - prevPos;
		}
	}

	public void setCollocationPrior(double a, double b) {
		this.collocationPrior = b;
		this.betaTotal = a + b;
	}

	public void setLMPrior(double prior) {
		this.lmPrior = prior;
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

	protected void takeSampleWithoutNullAlignments() {
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
			int prevLink = 0;

			for (int link = 0; link < alignmentVector.length; link++) {
				if (collocationVector[link]) {
					sentenceSamples.get(link).put(prevLink, 1.0);
				} else {
					sentenceSamples.get(link).put(alignmentVector[link], 1.0);
					prevLink = alignmentVector[link];
				}
			}
		}
	}

	protected int newRandomLink(int prevLink, int nextLink, int englishLength) {
		double[] alignmentProbs = new double[englishLength];
		double total = 0;
		for (int pos = 1; pos < englishLength; pos++) {
			double score = this.scoreTransition(prevLink, pos, nextLink);
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

	protected void removeCollocationInformation(int prevWord, int currentWord) {
		int count = (int) Math.round(this.continuations.get(prevWord));
		if (count != 0) {
			this.continuationCounts.subtract(count, 1.0);
			if (count != 1) {
				this.continuationCounts.put(count - 1, 1.0);
			}
		}

		this.continuations.subtract(prevWord, 1.0);
		this.collocationTable.subtract(prevWord, currentWord, 1.0);
	}

	protected void updateCollocationInformation(int prevWord, int currentWord) {
		// adjust conditional counts
		int count = (int) Math.round(this.continuations.get(prevWord));
		if (this.continuationCounts.containsKey(count)) {
			this.continuationCounts.subtract(count, 1.0);
		}
		this.continuationCounts.put(count + 1, 1.0);

		this.continuations.put(prevWord, 1.0);
		this.collocationTable.put(prevWord, currentWord, 1.0);
	}

	@Override
	protected void assignInitialAlignment(ParallelCorpus corpus) throws IOException {

		IBM1 baseModel;
		if (!this.ibm1Table.equals("")) {
			LogTranslationTable<Integer> translationTable = TranslationTableUtils
					.readNumericTranslationTable(this.ibm1Table);
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

			int jump = computeJump(1, englishSide.length - 1);
			this.maxJump = jump > this.maxJump ? jump : this.maxJump;

			int prevWord = this.SOURCE_START_SYMBOL;
			int prevLink = this.START_STATE;
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

					// randomly choose new link according to previously seen
					// links
					alignmentPoint = newRandomLink(j, frenchSide.length, englishSide.length);
				} else {
					int sourceCount = (int) Math
							.round(this.translationCountTable.get(englishSide[alignmentPoint], frenchSide[j]));
					if (this.conditionalSourceCounts.containsKey(sourceCount)) {
						this.conditionalSourceCounts.subtract(sourceCount, 1.0);
					}
					this.conditionalSourceCounts.put(sourceCount + 1, 1.0);

					this.translationCountTable.put(englishSide[alignmentPoint], frenchWord, 1.0);
					this.englishTotals.put(englishSide[alignmentPoint], 1.0);
				}
				currentAlignment[j] = alignmentPoint;
				this.transitionCounts.put(computeJump(prevLink, alignmentPoint), 1.0);
				prevLink = alignmentPoint;
				prevWord = frenchWord;
			}
			this.state.add(currentAlignment);
			this.collocationIndicators.add(collocationVector);
		}

		if (this.assymetricIBM1Prior) {
			setBaseDistribution(tTable.toRealSpace());
		}

		this.lmPriorTotal = this.sourceWordFreqs.size() * this.lmPrior;
		setTransitionPrior(this.transitionPrior.get(0));

		computeSourceTotals();

		// initialize indicators
		System.out.println("Starting to initialise collocation indicators at " + new Date());
		for (int iter = 1; iter <= 200; iter++) {
			sampleCollocationIndicatorsOnce(corpus);
		}
		System.out.println("Finished initialising collocation indicators at " + new Date());
	}

	private void computeSourceTotals() {
		for (double value : this.sourceWordFreqs.values()) {
			this.sourceContextTotals.put((int) Math.round(value), 1.0);
		}
	}

	protected void sampleCollocationIndicatorsOnce(ParallelCorpus corpus) {
		for (int sent = 0; sent < corpus.size(); sent++) {
			int[][] pair = corpus.get(sent);
			int[] frenchSide = pair[0];
			int frenchLength = frenchSide.length;
			int[] englishSide = pair[1];

			boolean[] collocationVector = this.collocationIndicators.get(sent);
			int[] alignmentVector = this.state.get(sent);

			int prevWord = this.SOURCE_START_SYMBOL;
			int prevLink = this.START_STATE;

			for (int j = 0; j < frenchLength; j++) {
				int nextLink = j + 1 < frenchSide.length ? alignmentVector[j + 1] : this.END_STATE;
				int fword = frenchSide[j];
				int currentLink = alignmentVector[j];
				boolean wasCollocated;

				if (wasCollocated = collocationVector[j]) {
					removeCollocationInformation(prevWord, fword);
				} else {
					this.removeInformation(prevLink, currentLink, nextLink, englishSide, j, frenchSide);
				}

				boolean newIndicator = sampleCollocationIndicator(prevLink, currentLink, englishSide, j - 1, j,
						frenchSide);

				if (newIndicator) {
					updateCollocationInformation(prevWord, fword);
					if (!wasCollocated) {
						this.transitionCounts.put(computeJump(prevLink, currentLink), 1.0);
						if (nextLink != this.END_STATE) {
							this.transitionCounts.put(computeJump(currentLink, nextLink), 1.0);
						}
					}
				} else if (wasCollocated) {
					// indicator has changed from collocation to alignment ->
					// thus no jump needs to be added
					int englishWord = englishSide[currentLink];

					int sourceCount = (int) Math.round(this.translationCountTable.get(englishWord, frenchSide[j]));
					if (this.conditionalSourceCounts.containsKey(sourceCount)) {
						this.conditionalSourceCounts.subtract(sourceCount, 1.0);
					}
					this.conditionalSourceCounts.put(sourceCount + 1, 1.0);

					this.translationCountTable.put(englishWord, frenchSide[j], 1.0);
					this.englishTotals.put(englishWord, 1.0);
				} else {
					this.updateInformation(prevLink, currentLink, nextLink, englishSide, j, frenchSide);
				}
				collocationVector[j] = newIndicator;
				prevWord = fword;
				prevLink = currentLink;
			}
		}
	}

	protected boolean sampleCollocationIndicator(int prevLink, int currentLink, int[] englishSide, int prevFrenchPos,
			int currentFrenchPos, int[] frenchSide) {
		int prevWord = prevFrenchPos == -1 ? this.SOURCE_START_SYMBOL : frenchSide[prevFrenchPos];
		int currentWord = frenchSide[currentFrenchPos];
		int alignedWord = englishSide[currentLink];

		double continuationProb = this.continuations.get(prevWord) + this.collocationPrior;
		double collocationProb = (this.collocationTable.get(prevWord, currentWord) + this.lmPrior)
				/ (this.continuations.get(prevWord) + this.lmPriorTotal);
		// double distortionProb = this.scoreTransition(prevLink, currentLink);
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

		int sizeOfSupport = sourceWordFreqs.size();
		int samplesTaken = 0;
		// an upper bound on the number of samples to be taken in total

		this.translationPriorTotal = baseDistribution == null ? childConcentration * sizeOfSupport : childConcentration;

		// stores the probabilities for no co-occurrences with a given total
		// count
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
			
			if (this.hyper && iter%lag == 0) {
				long hyperStart = System.nanoTime();
				sliceSampleHyperparameters(10);
				System.out.printf("Slice sampling took %f seconds%n", (System.nanoTime() - hyperStart) / 1000000000.0);
				// TODO probability cache does not need to be cleared when
				// hyperparameters for translations are not changed
				// probabilityCache.clear();
			}

			// take a sample
			if (iter >= burnIn && iter % lag == 0 && !this.importanceSampler) {
				if (this.alignAll) {
					this.takeSampleWithoutNullAlignments();
				} else {
					this.takeSample();
				}
				
				if (this.hyperAfterSample) {
					long hyperStart = System.nanoTime();
					sliceSampleHyperparameters(10);
					System.out.printf("Slice sampling took %f seconds%n", (System.nanoTime() - hyperStart) / 1000000000.0);
				}
				samplesTaken++;
			}

			if (this.experimentPrintOut > 0 && iter % this.experimentPrintOut == 0) {
				this.printAlignmentsDuringExperiment(iter);
			}

			if (this.printLikelihood) {
				// TODO
				System.out.printf("Log-Likelihood at iteration %d: %f%n", iter, this.logLikelihood);
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
			int prevLink = this.START_STATE;

			for (int j = 0; j < frenchSide.length; j++) {
				int[] competition = this.defineCompetition(englishSide, alignmentVector[j], englishPositions);
				int newLink = -1;
				int nextLink = j + 1 < frenchSide.length ? alignmentVector[j + 1] : this.END_STATE;

				if (collocationVector[j]) {
					int currentLink = alignmentVector[j];
					this.transitionCounts.subtract(computeJump(prevLink, currentLink), 1.0);
					if (nextLink != this.END_STATE) {
						this.transitionCounts.subtract(computeJump(currentLink, nextLink), 1.0);
					}

					double[] probs = new double[competition.length];
					double total = 0;
					for (int i = 0; i < competition.length; i++) {
						double score = this.scoreTransition(prevLink, competition[i], nextLink);
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

					this.transitionCounts.put(computeJump(prevLink, newLink), 1.0);
					if (nextLink != this.END_STATE) {
						this.transitionCounts.put(computeJump(newLink, nextLink), 1.0);
					}

					alignmentVector[j] = newLink;
				} else {
					int currentLink = alignmentVector[j];
					this.removeInformation(prevLink, currentLink, nextLink, englishSide, j, frenchSide);

					double[][] alignProbs = getAlignmentProbs(prevLink, nextLink, englishSide, competition, frenchSide,
							j, probabilityCache, translationPriorTotal);

					double[] probs = alignProbs[0];
					double totalProbs = alignProbs[1][0];

					double threshold = randomGenerator.nextDouble() * totalProbs;
					for (int item : competition) {
						if ((threshold -= probs[item]) <= 0) {
							newLink = item;
							break;
						}
					}
					this.updateInformation(prevLink, newLink, nextLink, englishSide, j, frenchSide);
					alignmentVector[j] = newLink;
				}
				prevLink = newLink;
			}
		}
	}

	@Override
	// ensure that NULL links never receive positive probability
	public double scoreTransition(int prevLink, int currentLink) {
		return currentLink == 0 ? 0 : super.scoreTransition(prevLink, currentLink);
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
	protected void sliceSampleHyperparameters(int iterations) {
		// super.sliceSampleHyperparameters(iterations);
		sampleCollocationHyperparameters(iterations);
	}

	private IntCounter computeFailureCounts() {
		IntCounter failureCounts = new IntCounter();
		// TODO account for start token
		for (int i = 1; i <= this.sourceWordFreqs.size(); i++) {
			failureCounts.put((int) Math.round(this.sourceWordFreqs.get(i) - this.continuations.get(i)), 1.0);
		}
		return failureCounts;
	}

	private void sampleCollocationHyperparameters(int iterations) {
		if (this.collocationHyperSampler == null) {
			this.collocationHyperSampler = new MultivariateSliceSampler<Integer>(new GammaDistribution(1, 1),
					new BinomialLikelihood(this.continuationCounts, computeFailureCounts(), this.sourceContextTotals));
		} else {
			this.collocationHyperSampler.setLikelihood(
					new BinomialLikelihood(this.continuationCounts, computeFailureCounts(), this.sourceContextTotals));
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

	private class BinomialLikelihood implements MultivariateLikelihood<Integer> {
		// TODO IMPORANT: beta is continuation hyperprior! Thus beta is prior on
		// successes

		private IntCounter successCounts;
		private IntCounter contextCounts;
		private IntCounter failureCounts;
		private double alpha;
		private double beta;
		private double likelihood;

		public BinomialLikelihood(IntCounter successCounts, IntCounter failureCounts, IntCounter contextCounts) {
			this.successCounts = successCounts;
			this.failureCounts = failureCounts;
			this.contextCounts = contextCounts;
			this.likelihood = 0;
		}

		@Override
		public double compute(Map<Integer, Double> parameter) throws IllegalArgumentException {
			if (!(2 == parameter.size())) {
				throw new IllegalArgumentException("Length of parameter vector has to be exactly 2!");
			}
			this.alpha = parameter.get(0);
			this.beta = parameter.get(1);

			likelihood = contextCounts.getTotal() * Gamma.logGamma(alpha + beta);
			// only need to subtract for those events that have success/failure
			// -> all others cancel
			likelihood -= failureCounts.getTotal() * Gamma.logGamma(alpha);
			likelihood -= successCounts.getTotal() * Gamma.logGamma(beta);

			for (Map.Entry<Integer, Double> entry : this.successCounts.entrySet()) {
				likelihood += Gamma.logGamma(beta + entry.getKey()) * entry.getValue();
			}
			for (Map.Entry<Integer, Double> entry : this.failureCounts.entrySet()) {
				likelihood += Gamma.logGamma(alpha + entry.getKey()) * entry.getValue();
			}
			for (Map.Entry<Integer, Double> entry : this.contextCounts.entrySet()) {
				likelihood -= Gamma.logGamma(alpha + beta + entry.getKey()) * entry.getValue();
			}

			return likelihood;
		}

		@Override
		public double computeAt(Integer dimension, double value) {
			if (dimension < 0 || dimension > 2) {
				// TODO throw Exception
			}

			double result = this.likelihood;

			if (dimension == 0) {
				result += contextCounts.getTotal() * (Gamma.logGamma(value + beta) - Gamma.logGamma(alpha + beta));
				result -= failureCounts.getTotal() * (Gamma.logGamma(value) - Gamma.logGamma(alpha));
				for (Map.Entry<Integer, Double> entry : this.failureCounts.entrySet()) {
					result += (Gamma.logGamma(value + entry.getKey()) - Gamma.logGamma(this.alpha + entry.getKey()))
							* entry.getValue();
				}
				for (Map.Entry<Integer, Double> entry : this.contextCounts.entrySet()) {
					result -= Gamma.logGamma(value + beta + entry.getKey()) * entry.getValue();
				}
			} else {
				result += contextCounts.getTotal() * (Gamma.logGamma(alpha + value) - Gamma.logGamma(alpha + beta));
				result -= successCounts.getTotal() * (Gamma.logGamma(value) - Gamma.logGamma(beta));
				for (Map.Entry<Integer, Double> entry : this.successCounts.entrySet()) {
					result += (Gamma.logGamma(value + entry.getKey()) - Gamma.logGamma(this.beta + entry.getKey()))
							* entry.getValue();
				}
				for (Map.Entry<Integer, Double> entry : this.contextCounts.entrySet()) {
					result -= (Gamma.logGamma(alpha + value + entry.getKey())
							- Gamma.logGamma(alpha + beta + entry.getKey())) * entry.getValue();
				}
			}

			return result;
		}

		@Override
		public void updateAt(Integer dimension, double value) {
			if (dimension < 0 || dimension > 2) {
				// TODO throw Exception
			}

			if (dimension == 0) {
				this.likelihood += contextCounts.getTotal() * (Gamma.logGamma(value + beta) - Gamma.logGamma(alpha + beta));
				this.likelihood -= failureCounts.getTotal() * (Gamma.logGamma(value) - Gamma.logGamma(alpha));
				for (Map.Entry<Integer, Double> entry : this.failureCounts.entrySet()) {
					this.likelihood += (Gamma.logGamma(value + entry.getKey()) - Gamma.logGamma(this.alpha + entry.getKey()))
							* entry.getValue();
				}
				for (Map.Entry<Integer, Double> entry : this.contextCounts.entrySet()) {
					this.likelihood -= Gamma.logGamma(value + beta + entry.getKey()) * entry.getValue();
				}
				this.alpha = value;
			} else {
				this.likelihood += contextCounts.getTotal() * (Gamma.logGamma(alpha + value) - Gamma.logGamma(alpha + beta));
				this.likelihood -= successCounts.getTotal() * (Gamma.logGamma(value) - Gamma.logGamma(beta));
				for (Map.Entry<Integer, Double> entry : this.successCounts.entrySet()) {
					this.likelihood += (Gamma.logGamma(value + entry.getKey()) - Gamma.logGamma(this.beta + entry.getKey()))
							* entry.getValue();
				}
				for (Map.Entry<Integer, Double> entry : this.contextCounts.entrySet()) {
					this.likelihood -= (Gamma.logGamma(alpha + value + entry.getKey())
							- Gamma.logGamma(alpha + beta + entry.getKey())) * entry.getValue();
				}
				this.beta = value;
			}
		}

		@Override
		public boolean hasObservation(Integer dimension) {
			return true;
		}
	}
}
