package BayesianIBM;

import ibmModels.IBM1;
import ibmModels.VBIBM1;
import io.GizaUtils;
import io.TranslationTableUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.special.Gamma;

import base.BayesAlign;
import alignmentUtils.ParallelCorpus;
import collections.Counter;
import collections.DoubleCounter;
import collections.IntCountTable;
import collections.IntCounter;
import collections.Pair;
import collections.machineTranslation.LogTranslationTable;
import collections.machineTranslation.TranslationTable;
import sampling.HyperparameterSliceSampler;
import sampling.Likelihood;

/**
 * This class implements a Bayesian IBM1 aligner as described in Mermer et al. (2010). Inference is done using Gibbs sampling.
 * 
 * @author Philip Schulz
 *
 */
public class GibbsIBM1 extends BayesAlign {

	protected IntCountTable translationCountTable;
	protected IntCounter englishTotals;
	protected double translationPriorTotal;
	// List of samples from sentences to French positions

	protected boolean assymetricIBM1Prior;
	protected TranslationTable<Integer> baseDistribution;

	// this is for to print out alignments along the way in experiments
	protected int experimentPrintOut;
	protected String experimentFormat;
	protected boolean ttable;
	protected String ibm1Table;

	protected boolean printLikelihood;

	protected boolean importanceSampler;
	protected boolean mhSampler;
	protected List<double[][]> proposalDist;

	// profiling

	// necessary for slice sampling hyperparameters
	protected IntCounter conditionalSourceCounts;
	protected HyperparameterSliceSampler translationHyperSampler;
	protected GammaDistribution translationHyperprior;

	public GibbsIBM1(double translationPrior) {
		this();
		setTranslationPrior(translationPrior);
	}

	public GibbsIBM1() {
		this.randomGenerator = new MersenneTwister();
		this.state = new ArrayList<int[]>();
		// stores counts of English-to-French co-occurrences
		this.translationCountTable = new IntCountTable();
		this.samples = new ArrayList<List<IntCounter>>();
		this.englishTotals = new IntCounter();
		// TODO at the moment gibbs-aux is default sampler
		this.aux = true;
		this.ibm1Table = "";
		this.experimentPrintOut = 0;
		this.printLikelihood = false;
		this.importanceSampler = false;
		this.mhSampler = false;
		this.ttable = false;
		this.assymetricIBM1Prior = false;
		this.baseDistribution = null;
		this.conditionalSourceCounts = new IntCounter();

		// TODO take this out later on
		// TODO integrate this with the sampler -> the actual gamma parameters
		// may not need to be stored
		this.translationGammaA = 1;
		this.translationGammaB = 0.1;
		this.translationHyperprior = new GammaDistribution(this.translationGammaA, 1 / this.translationGammaB);
	}

	public void setTranslationPrior(double prior) {
		this.childConcentration = prior;
	}

	public double getTranslationPrior() {
		return this.childConcentration;
	}

	public void initializeIBM1FromTable(String pathToFile) {
		this.ibm1Table = pathToFile;
	}

	public void setExperimentPrintOut(int value) {
		this.experimentPrintOut = value;
	}

	public void setTTable(boolean value) {
		this.ttable = value;
	}

	public void setAux(boolean value) {
		this.aux = value;
	}

	public void printLikelihood() {
		this.printLikelihood = true;
	}

	/**
	 * Use the IBM1 translation table in order to induce an assymetric Dirichlet prior. As a consequence the Dirichlet parameters will be
	 * set to translationPrior*IBM1table.
	 */
	public void useIBM1TableForTranslationPrior() {
		this.assymetricIBM1Prior = true;
	}

	private void setSampler(boolean mh, boolean is) {
		this.mhSampler = mh;
		this.importanceSampler = is;
	}

	public void useMHSampler() {
		setSampler(true, false);
	}

	public void useImportanceSampler() {
		setSampler(false, true);
		this.samples = new ArrayList<List<IntCounter>>();
	}

	public void useGibbsSampler() {
		setSampler(false, false);
	}

	public TranslationTable<Integer> getBaseDistribution() {
		return this.baseDistribution;
	}

	public void setBaseDistribution(TranslationTable<Integer> distribution) {
		this.baseDistribution = distribution;
	}

	@Override
	public void align(String pathToSNTFile, String outputFile, String format, int iterations, int burnIn, int lag)
			throws FileNotFoundException, IOException {
		this.experimentFormat = format;
		ParallelCorpus corpus = ParallelCorpus.readCorpus(pathToSNTFile);
		assignInitialAlignment(corpus);

		this.sample(corpus, iterations, burnIn, lag);

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
			writeAlignments(writer, format);
		}
	}

	/**
	 * Train an IBM1 model and use its Viterbi alignment in order to initialize the state of the Markov chain. In case
	 * 
	 * 
	 * @param corpus
	 *            A numeric representation of the corpus stored in memory
	 * @throws FileNotFoundException
	 *             If path is wrong
	 * @throws IOException
	 */
	protected void assignInitialAlignment(ParallelCorpus corpus) throws IOException {
		this.sizeOfSupport = (int) Math.round(this.countSourceWords(corpus).getTotal());
		
		IBM1 baseModel;
		if (!this.ibm1Table.equals("")) {
			LogTranslationTable<Integer> translationTable = TranslationTableUtils.readNumericTranslationTable(this.ibm1Table);
			baseModel = IBM1.initializeWith(translationTable);
		} else {
			baseModel = VBIBM1.createModel(corpus, 5);
			baseModel.writeTranslationTableToFile("translation_table_ibm1");
		}
		LogTranslationTable<Integer> tTable = baseModel.getLogTranslationTable();
		if (this.importanceSampler || this.mhSampler) {
			this.proposalDist = baseModel.allLinkPosteriors(corpus);
		}

		for (int[][] sentencePair : corpus) {
			int[] frenchSide = sentencePair[0];
			int[] englishSide = sentencePair[1];
			int[] currentAlignment = new int[frenchSide.length];

			for (int j = 0; j < frenchSide.length; j++) {
				int frenchWord = frenchSide[j];
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
				} else {
					updateInformation(alignmentPoint, englishSide, j, frenchSide, currentAlignment);
				}
			}
			this.state.add(currentAlignment);
		}
	}

	/**
	 * Uses Gibbs sampling to re-sample alignment links for French words. A Dirichlet prior must be supplied in order to control which
	 * distributions will be most influential on the sampler.
	 * 
	 * @param corpus
	 *            A numeric representation of the corpus stored in memory
	 * @param iterations
	 *            The number of iterations that the Gibbs sampler will perform
	 * @param burnIn
	 *            The number of initial iterations for which no samples will be taken
	 * @param lag
	 *            The lag between individual samples
	 * @throws FileNotFoundException
	 *             If the path to the file is incorrect
	 * @throws IOException
	 */
	protected void sample(ParallelCorpus corpus, int iterations, int burnIn, int lag) throws IOException {
		double decile = Math.floor(iterations / 10.0);
		HashMap<Integer, int[]> positionsPerSentenceLength = new HashMap<Integer, int[]>();

		// TODO importance sampler never uses the auxiliary variable
		if (this.importanceSampler) {
			setAux(false);
		}

		int samplesTaken = 0;
		// an upper bound on the number of samples to be taken in total

		this.translationPriorTotal = baseDistribution == null ? childConcentration * sizeOfSupport : childConcentration;

		// stores the probabilities for no co-occurrences with a given total
		// count
		DoubleCounter probabilityCache = new DoubleCounter();
		probabilityCache.put(0.0, childConcentration / this.translationPriorTotal);

		DoubleCounter importanceCache = new DoubleCounter();

		System.out.println("Started sampling at " + new Date());

		for (int iter = 1; iter <= iterations; iter++) {
			if (iter % decile == 0) {
				System.out.println("Iteration " + iter);
			}

			if (this.mhSampler) {
				// TODO mhSampleOnce(corpus, translationPriorTotal,
				// probabilityCache);
			} else if (this.importanceSampler) {
				importanceSampleOnce(corpus, translationPriorTotal, importanceCache, positionsPerSentenceLength);
			} else {
				long gibbStart = System.nanoTime();
				gibbsSampleOnce(corpus, positionsPerSentenceLength, translationPriorTotal, probabilityCache, baseDistribution);
				System.out.printf("Gibbs iteration took %f seconds%n", (System.nanoTime() - gibbStart) / 1000000000.0);

				// TODO
				if (hyper && iter % (4 * lag) == 0) {

				}
			}

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

		if (this.importanceSampler) {
			samplesTaken = iterations;
		}
		System.out.println("Finished sampling at " + new Date());
		System.out.printf("%d samples taken in total.%n", samplesTaken);
	}

	/**
	 * Sample a new alignment configuration using Metropolis-Hastings.
	 * 
	 * @param corpus
	 *            The corpus to be aligned
	 * @param translationPriorTotal
	 *            The sum of the Dirichlet hyperparameters
	 * @param probabilityCache
	 *            A cache storing the posterior predictive probabilities for certain counts
	 */
	protected void mhSampleOnce(ParallelCorpus corpus, double translationPriorTotal, DoubleCounter probabilityCache) {
		int sentencePair = 0;

		for (int[][] nextPair : corpus) {
			int[] alignmentVector = this.state.get(sentencePair);
			double[][] proposals = this.proposalDist.get(sentencePair);
			int[] frenchSide = nextPair[0];
			int[] englishSide = nextPair[1];

			for (int frenchPos = 0; frenchPos < frenchSide.length; frenchPos++) {
				double[] currentProposals = proposals[frenchPos];
				int currentLink = alignmentVector[frenchPos];
				double currentLinkProposalScore = currentProposals[currentLink];

				Pair<Integer, Double> linkProposal = drawProposal(currentProposals);
				int proposedLink = linkProposal._0;
				double proposalScore = linkProposal._1;

				this.removeInformation(currentLink, englishSide, frenchPos, frenchSide);
				int[] competition = new int[] { currentLink, proposedLink };
				double[] alignmentScores = this.getAlignmentProbs(englishSide, competition, frenchSide, frenchPos, probabilityCache,
						translationPriorTotal)[0];
				double threshold = (alignmentScores[proposedLink] * currentLinkProposalScore)
						/ (alignmentScores[currentLink] * proposalScore);
				int newLink;
				if (threshold >= 1) {
					newLink = proposedLink;
				} else {
					newLink = (this.randomGenerator.nextDouble() < threshold) ? proposedLink : currentLink;
				}
				this.updateInformation(newLink, englishSide, frenchPos, frenchSide, alignmentVector);
			}
			sentencePair++;
		}
	}

	/**
	 * Sample a new alignment configuration using Metropolis-Hastings.
	 * 
	 * @param corpus
	 *            The corpus to be aligned
	 * @param translationPriorTotal
	 *            The sum of the Dirichlet hyperparameters
	 * @param probabilityCache
	 * @param positionsPerSentenceLength
	 *            A map from sentence lengths to integer arrays containing their positions A cache storing the posterior predictive
	 *            probabilities for certain counts ratios
	 */
	protected void importanceSampleOnce(ParallelCorpus corpus, double translationPriorTotal, DoubleCounter probabilityCache,
			HashMap<Integer, int[]> positionsPerSentenceLength) {
		int sentencePair = 0;
		double proposalScore = 0;

		List<int[]> proposal = new ArrayList<int[]>();
		for (int[][] nextPair : corpus) {
			double[][] proposals = this.proposalDist.get(sentencePair);

			int[] frenchSide = nextPair[0];
			int[] englishSide = nextPair[1];
			int[] alignment = new int[frenchSide.length];

			for (int frenchPos = 0; frenchPos < frenchSide.length; frenchPos++) {
				Pair<Integer, Double> linkProposal = drawProposal(proposals[frenchPos]);
				proposalScore += Math.log(linkProposal._1);
				this.updateInformation(linkProposal._0, englishSide, frenchPos, frenchSide, alignment);
			}
			proposal.add(alignment);
			sentencePair++;
		}

		double modelScore = computeLogLikelihood();
		double importanceWeight = modelScore - proposalScore;
		System.out.println("Importance weight = " + importanceWeight);
		// TODO create LogIntCounter to reactive this snippet
//		if (this.samples.isEmpty()) {
//
//			// initialize list of samples per sentence
//			for (int[] sent : this.state) {
//				List<Counter<Integer>> sentence = new ArrayList<Counter<Integer>>();
//				for (@SuppressWarnings("unused")
//				int link : sent) {
//					sentence.add(new LogCounter<Integer>());
//				}
//				this.samples.add(sentence);
//			}
//		}

		for (int sent = 0; sent < proposal.size(); sent++) {
			int[] sample = proposal.get(sent);
			List<IntCounter> samples = this.samples.get(sent);
			for (int j = 0; j < sample.length; j++) {
				// the samples data structure is a LogCounter when importance
				// sampling is used
				samples.get(j).put(sample[j], importanceWeight);
			}
		}
		
		this.translationCountTable.clear();
		this.conditionalSourceCounts.clear();
		this.englishTotals.clear();
	}

	/**
	 * Generate a proposal for an alingment link from the proposal distribution.
	 * 
	 * @param proposalDist
	 *            The proposalDistribution for this link
	 * @return A pair containing the proposed alignment position and the link probability according to the proposal distribution.
	 */
	protected Pair<Integer, Double> drawProposal(double[] proposalDist) {
		double threshold = randomGenerator.nextDouble();
		int pos = 0;
		double prob = proposalDist[pos];
		double total = prob;

		while (total < threshold) {
			pos++;
			prob = proposalDist[pos];
			total += prob;
		}
		return Pair.buildPairFrom(pos, prob);
	}

	/**
	 * Sample a new alignment configuration using Gibbs sampling.
	 * 
	 * @param corpus
	 *            The corpus to be aligned
	 * @param postitionsPerSentenceLength
	 *            A map from sentence lengths to vectors storing the word positions for each length
	 * @param translationPriorTotal
	 *            The sum of the Dirichlet hyperparameters
	 * @param probabilityCache
	 *            A cache storing the posterior predictive probabilities for certain counts
	 * @param baseDistribution
	 *            A base distribution that can be used to induce non-symmetric Dirichlet hyperparemters. If null is supplied as a value, a
	 *            symmetric hyperparameters will be used.
	 */
	// TODO make probability cache and prior total into fields
	protected void gibbsSampleOnce(ParallelCorpus corpus, HashMap<Integer, int[]> positionsPerSentenceLength, double translationPriorTotal,
			DoubleCounter probabilityCache, TranslationTable<Integer> baseDistribution) {
		int sentencePair = 0;
		for (int[][] nextPair : corpus) {
			int[] frenchSide = nextPair[0];
			int[] englishSide = nextPair[1];

			int[] englishPositions = getEnglishPositions(englishSide.length, positionsPerSentenceLength);

			int[] alignment = this.state.get(sentencePair);

			for (int j = 0; j < alignment.length; j++) {
				int currentAlignedPos = alignment[j];

				// define the competition for the current link
				int[] competition = defineCompetition(englishSide, currentAlignedPos, englishPositions);

				// remove all information obtained from this alignment link
				this.removeInformation(currentAlignedPos, englishSide, j, frenchSide);

				double[][] alignProbs = getAlignmentProbs(englishSide, competition, frenchSide, j, probabilityCache, translationPriorTotal);
				double[] probs = alignProbs[0];
				double totalProbs = alignProbs[1][0];

				// scale the sampled threshold to the alignment probabilities
				// this relieves us of having to scale each alignment prob
				double threshold = randomGenerator.nextDouble() * totalProbs;

				double accumulator = 0;
				int alignmentPosition = 0;
				double lastProb = 0;
				for (int item : competition) {
					alignmentPosition = item;
					lastProb = probs[alignmentPosition];
					accumulator += lastProb;

					if (accumulator >= threshold) {
						break;
					}
				}

				// update with information obtained from new alignment position
				updateInformation(alignmentPosition, englishSide, j, frenchSide, alignment);
			}
			sentencePair++;
		}
	}

	protected int[] getEnglishPositions(int sentenceLength, Map<Integer, int[]> positionsPerSentenceLength) {
		int[] englishPositions;
		if (positionsPerSentenceLength.containsKey(sentenceLength)) {
			englishPositions = positionsPerSentenceLength.get(sentenceLength);
		} else {
			englishPositions = new int[sentenceLength];
			for (int i = 0; i < sentenceLength; i++) {
				englishPositions[i] = i;
			}
			positionsPerSentenceLength.put(sentenceLength, englishPositions);
		}
		return englishPositions;
	}

	/**
	 * Update the information after a new alignment link has been sampled.
	 * 
	 * @param englishPos
	 *            The newly sampled English position
	 * @param englishSide
	 *            The English words in order
	 * @param frenchPos
	 *            The French position of the alignment link
	 * @param frenchSide
	 *            The French words in order
	 * @param alignmentVector
	 *            the alignment vector to be updated
	 */
	protected void updateInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide, int[] alignmentVector) {
		int englishWord = englishSide[englishPos];
		int frenchWord = frenchSide[frenchPos];

		// adjust conditional counts
		int sourceCount = (int) Math.round(this.translationCountTable.get(englishWord, frenchWord));
		if (this.conditionalSourceCounts.containsKey(sourceCount)) {
			this.conditionalSourceCounts.subtract(sourceCount, 1.0);
		}
		this.conditionalSourceCounts.put(sourceCount + 1, 1.0);

		this.translationCountTable.put(englishWord, frenchWord, 1.0);
		this.englishTotals.put(englishWord, 1.0);
		alignmentVector[frenchPos] = englishPos;
	}

	/**
	 * Removes information about the current alignment from the sampler
	 * 
	 * @param englishPos
	 *            The currently aligned English position
	 * @param englishSide
	 *            The English words in order
	 * @param frenchPos
	 *            The currently aligned French position
	 * @param frenchSide
	 *            The French words in order
	 */
	protected void removeInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide) {
		int englishWord = englishSide[englishPos];
		int frenchWord = frenchSide[frenchPos];

		int sourceCount = (int) Math.round(this.translationCountTable.get(englishWord, frenchWord));
		this.conditionalSourceCounts.subtract(sourceCount, 1.0);
		if (sourceCount != 1) {
			this.conditionalSourceCounts.put(sourceCount - 1, 1.0);
		}

		this.translationCountTable.subtract(englishWord, frenchWord, 1.0);
		this.englishTotals.subtract(englishWord, 1.0);
	}

	/**
	 * Compute the alignment probability with the current French word for each English word in the competition.
	 * 
	 * @param englishSide
	 *            The English words of the sentence pair in order
	 * @param competition
	 *            The competition of English positions
	 * @param frenchSide
	 *            The French words of the sentence pair in order
	 * @param frenchPos
	 *            The position of the current French word
	 * @param probabilityCache
	 *            A cache storing previously computed alignment probabilities per number of observations
	 * @param translationPriorTotal
	 *            The sum of the Dirichlet hyperparameters over the entire vocabulary
	 * @return An array containing an array of alignment probabilities for each element in the competition in the first slot and a singleton
	 *         array containing the total of those probabilities in the second slot
	 */
	protected double[][] getAlignmentProbs(int[] englishSide, int[] competition, int[] frenchSide, int frenchPos,
			DoubleCounter probabilityCache, double translationPriorTotal) {
		double[] probs = new double[englishSide.length];
		double totalProbs = 0;
		int frenchWord = frenchSide[frenchPos];

		// compute alignment probs
		for (int i : competition) {
			int englishWord = englishSide[i];
			double count = this.translationCountTable.get(englishWord, frenchWord);
			double total = this.englishTotals.get(englishWord);

			double alignmentProb = scoreTranslationPair(englishWord, frenchWord, count, total, translationPriorTotal, probabilityCache);
			probs[i] = alignmentProb;
			totalProbs += alignmentProb;
		}

		return new double[][] { probs, new double[] { totalProbs } };
	}

	public double scoreTranslationPair(int englishWord, int frenchWord, double count, double total, double translationPriorTotal,
			DoubleCounter probabilityCache) {
		if (this.baseDistribution == null) {
			return getProbabilityFromSymmetricPrior(englishWord, frenchWord, translationPriorTotal, probabilityCache);
		} else {
			return getProbabilityFromAsymetricPrior(englishWord, frenchWord, count, total, translationPriorTotal);
		}
	}

	protected double getProbabilityFromSymmetricPrior(int englishWord, int frenchWord, double translationPriorTotal,
			DoubleCounter probabilityCache) {
		double count = this.translationCountTable.get(englishWord, frenchWord);
		double total = this.englishTotals.get(englishWord);

		double alignmentProb;
		if (count == 0.0) {
			if (probabilityCache.containsKey(total)) {
				alignmentProb = probabilityCache.get(total);
			} else {
				alignmentProb = childConcentration / (this.translationPriorTotal + total);
				probabilityCache.put(total, alignmentProb);
			}
		} else {
			alignmentProb = (count + childConcentration) / (this.translationPriorTotal + total);
		}
		return alignmentProb;
	}

	protected double getProbabilityFromAsymetricPrior(int englishWord, int frenchWord, double count, double total,
			double translationPriorTotal) {
		return (count + this.childConcentration * this.baseDistribution.scoreTranslationPair(frenchWord, englishWord))
				/ (total + translationPriorTotal);
	}

	/**
	 * Get the competition of English words that may align with the current French word. The currently aligned English word is always part
	 * of the competition. If the auxiliary variable is used, only one other English word is sampled uniformly into the competition.
	 * Otherwise, the entire English sentence is used as competition.
	 * 
	 * @param englishSide
	 *            The English words of the sentence pair in order
	 * @param currentAlignedPos
	 *            The position of the currently aligned English word
	 * @param englishPositions
	 *            An array of all English positions
	 * @return The competition for alignment with the current French word (an integer array of English positions)
	 */
	protected int[] defineCompetition(int[] englishSide, int currentAlignedPos, int[] englishPositions) {
		int[] competition;

		if (this.aux) {
			int competitor = randomGenerator.nextInt(englishSide.length);
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

	protected void printAlignmentsDuringExperiment(int iter) throws IOException {
		File file = new File("experiment-alignments/alignments" + iter);
		file.getParentFile().mkdirs();

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
			writeAlignments(writer, this.experimentFormat);
		}
	}

	/**
	 * Write a lexical translation table to file. This file can be used to for lexical scoring in the Moses decoder to whose format it
	 * adheres.
	 * 
	 * @param corpus
	 *            The input corpus from which the alignments previously been sampled
	 * @param fileName
	 *            The file to which the translationTable shall be written
	 * @param pathToFrenchVCB
	 *            The path to the source .vcb file
	 * @param pathToEnglishVCB
	 *            The path to the target .vcb file
	 * @throws FileNotFoundException
	 *             If one of the .vcb files is not present
	 * @throws IOException
	 */
	private void writeTranslationTableToFile(ParallelCorpus corpus, String fileName, String pathToFrenchVCB, String pathToEnglishVCB)
			throws FileNotFoundException, IOException {
		HashMap<Integer, String> frenchWords = GizaUtils.readVCB(pathToFrenchVCB);
		HashMap<Integer, String> englishWords = GizaUtils.readVCB(pathToEnglishVCB);
		TranslationTable<Integer> translationTable = buildTranslationTableFromSamples(corpus);

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
			for (Map.Entry<Integer, Counter<Integer>> entry : translationTable.entrySet()) {
				String englishWord = entry.getKey() == 0 ? "NULL" : englishWords.get(entry.getKey());
				for (Map.Entry<Integer, Double> translation : entry.getValue().entrySet()) {
					String frenchWord = frenchWords.get(translation.getKey());
					writer.write(String.format("%s %s %f%n", englishWord, frenchWord, translation.getValue()));
				}
			}
		}
	}

	/*
	 * Construct a translation table from the samples. To this end the number of times that an English word has been aligned to a specific
	 * French word is averaged over the number of samples.
	 */
	protected TranslationTable<Integer> buildTranslationTableFromSamples(ParallelCorpus corpus) {
		TranslationTable<Integer> translationTable = TranslationTable.buildTable();
		for (int sent = 0; sent < this.samples.size(); sent++) {
			List<IntCounter> sentenceSamples = this.samples.get(sent);
			int[][] pair = corpus.get(sent);
			int[] sourceSide = pair[0];
			int[] targetSide = pair[1];

			for (int sourcePos = 0; sourcePos < sourceSide.length; sourcePos++) {
				IntCounter counts = sentenceSamples.get(sourcePos);
				int source = sourceSide[sourcePos];

				for (Map.Entry<Integer, Double> entry : counts.entrySet()) {
					int target = targetSide[entry.getKey()];
					translationTable.put(source, target, entry.getValue());
				}
			}
		}
		return translationTable;
	}

	public List<int[]> getState() {
		return state;
	}

	public IntCountTable getTranslationCountTable() {
		return translationCountTable;
	}

	public IntCounter getEnglishTotals() {
		return englishTotals;
	}

	public List<List<IntCounter>> getSamples() {
		return samples;
	}

	/**
	 * Computes the model log-likelihood with respect to the current sample.
	 * 
	 * @return The log-likelihood of the model under the current sample
	 */
	public double computeLogLikelihood() {
		Likelihood likelihood = new TranslationLikelihood(this.englishTotals.size(), getEnglishTotals(), this.conditionalSourceCounts,
				this.sizeOfSupport);
		if (this.hyper) {
			return likelihood.compute(this.translationPriorTotal) + this.translationHyperprior.logDensity(this.translationPriorTotal);
		} else {
			return likelihood.compute(this.translationPriorTotal);
		}
	}

	// //////////////////////// Hyperparameter Inference
	// //////////////////////////

	protected void sliceSampleHyperparameters(int iterations) {

		if (this.translationHyperSampler == null) {
			this.translationHyperSampler = new HyperparameterSliceSampler(new GammaDistribution(this.translationGammaA,
					1 / this.translationGammaB), new TranslationLikelihood(this.englishTotals.size(), getEnglishTotals(),
					this.conditionalSourceCounts, this.sizeOfSupport));
		} else {
			this.translationHyperSampler.setLikelihood(new TranslationLikelihood(this.englishTotals.size(), getEnglishTotals(),
					this.conditionalSourceCounts, this.sizeOfSupport));
		}

		this.translationHyperSampler.sample(this.translationPriorTotal, iterations, 0, 0);
		double newValue = this.translationHyperSampler.getState();

		this.translationPriorTotal = newValue;
		this.childConcentration = newValue / this.sizeOfSupport;

		System.out.println("alpha = " + this.childConcentration);
	}

	protected class TranslationLikelihood implements Likelihood {

		private int contextSize;
		private IntCounter contextCounts;
		private IntCounter observationCounts;
		private int sizeOfSupport;

		public TranslationLikelihood(int contextSize, IntCounter contextCounts, IntCounter observationCounts, int sizeOfSupport) {
			this.contextCounts = contextCounts;
			this.sizeOfSupport = sizeOfSupport;
			this.observationCounts = observationCounts;
			this.contextSize = contextSize;
		}

		@Override
		public double compute(double parameter) {
			double uniform = parameter / this.sizeOfSupport;

			double constants = contextSize * Gamma.logGamma(parameter);
			double sum = 0;
			for (Map.Entry<Integer, Double> entry : contextCounts.entrySet()) {
				sum -= Gamma.logGamma(entry.getValue() + parameter);
			}
			for (Map.Entry<Integer, Double> entry : this.observationCounts.entrySet()) {
				sum += entry.getValue() * (Gamma.logGamma(entry.getKey() + uniform) - Gamma.logGamma(uniform));
			}
			return constants + sum;
		}
	}
}
