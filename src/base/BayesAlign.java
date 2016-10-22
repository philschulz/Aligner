package base;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;

import alignmentUtils.ParallelCorpus;
import collections.IntCounter;
import sampling.Likelihood;

public abstract class BayesAlign {

	protected List<int[]> state;
	protected List<List<IntCounter>> samples;
	protected MersenneTwister randomGenerator;
	protected boolean aux;
	protected double childConcentration;
	protected Likelihood logLikelihood;
	protected int sizeOfSupport;

	// hyperparameters
	protected double translationGammaA;
	protected double translationGammaB;
	protected boolean hyper;

	public void setTranslationGammaParameters(double shape, double rate) {
		this.translationGammaA = shape;
		this.translationGammaB = rate;
	}

	/**
	 * Enables hyperparameter inference in the model.
	 */
	public void doHyperparameterInference() {
		this.hyper = true;
	}

	/**
	 * Return the translationPrior specified by the client.
	 * 
	 * @return The translationPrior
	 */
	public double getTranslationPrior() {
		return childConcentration;
	}

	/**
	 * Determine whether the Gibbs sampler uses an auxiliary variable to limit the number of transitions into new states.
	 * 
	 * @param aux
	 *            Gibbs sampler uses auxiliary variable if true
	 */
	public void setAux(boolean aux) {
		this.aux = aux;
	}

	/**
	 * Get the state of the Markov Chain sampler.
	 * 
	 * @return The state of the Markov chain
	 */
	public List<int[]> getState() {
		return this.state;
	}

	/**
	 * Count the word types on the source side of the bilingual corpus.
	 * 
	 * @param corpus
	 *            A bilingual corpus stored in memory
	 * @return A counter for the source words
	 * @throws IOException
	 */
	protected IntCounter countSourceWords(ParallelCorpus corpus) {
		IntCounter sourceWords = new IntCounter();

		for (int[][] sentencePair : corpus) {
			int[] frenchSide = sentencePair[0];

			for (Integer word : frenchSide) {
				sourceWords.put(word, 1.0);
			}
		}
		return sourceWords;
	}

	/**
	 * Align a parallel corpus using the Bayesian IBM1 model.
	 * 
	 * @param pathToSNTFile
	 *            Path to the corpus in .snt format
	 * @param outputFile
	 *            The file to which the alignments should be printed
	 * @param format
	 *            The format of the resulting alignments. Has to be one of {moses, naacl, talp}
	 * @param iterations
	 *            The number of iterations for which to run the Gibbs sampler
	 * @param burnIn
	 *            The number of initial iterations in which no samples are taken
	 * @param lag
	 *            The number of iterations between subsequent samples
	 * @throws FileNotFoundException
	 *             If the input file cannot be located on disk
	 * @throws IOException
	 */
	public abstract void align(String pathToSNTFile, String outputFile, String format, int iterations, int burnIn, int lag)
			throws FileNotFoundException, IOException;

	/**
	 * Write the most frequently sampled alignment links to disk
	 * 
	 * @param writer
	 *            A stream to the file that should be written to
	 * @param format
	 *            The format in which the alignments should be written
	 * @throws IOException
	 */
	protected void writeAlignments(BufferedWriter writer, String format) throws IOException {
		int lineNum = 1;
		for (List<IntCounter> sentence : this.samples) {
			for (int j = 0; j < sentence.size(); j++) {
				int bestLink = sentence.get(j).getHighestKey();
				if (bestLink == 0) {
					continue;
				} else if (format.equals("moses")) {
					// this needs to be sourceL-targetL in order to comply with Moses format
					writer.write(j + "-" + (bestLink - 1) + " ");
				} else if (format.equals("talp")) {
					// talp is identical to Moses but the indeces are shifted by one
					writer.write((j + 1) + "-" + bestLink + " ");
				} else if (format.equals("naacl")) {
					// naacl format (see Mihalcea and Pedersen, 2003)
					writer.write(String.format("%04d", lineNum) + " " + (j + 1) + " " + bestLink);
					writer.newLine();
				}
			}
			if (!format.equals("naacl")) {
				writer.newLine();
			}
			lineNum++;
		}
	}

	/**
	 * Get the competition of English words that may align with the current French word. The currently aligned English word is always part
	 * of the competition. If the auxiliary variable is used, only one other English word is sampled uniformly into the competition.
	 * Otherwise, the entire English sentence is used as competition.
	 * 
	 * @param targetSide
	 *            The English words of the sentence pair in order
	 * @param currentAlignedPos
	 *            The position of the currently aligned English word
	 * @param targetPositions
	 *            An array of all English positions
	 * @return The competition for alignment with the current French word (an integer array of English positions)
	 */
	protected int[] defineCompetition(int[] targetSide, int currentAlignedPos, int[] targetPositions) {
		int[] competition;

		if (this.aux) {
			int competitor = randomGenerator.nextInt(targetSide.length);
			if (competitor == currentAlignedPos) {
				return defineCompetition(targetSide, currentAlignedPos, targetPositions);
			} else {
				competition = new int[] { currentAlignedPos, competitor };
			}
		} else {
			competition = targetPositions;
		}

		return competition;
	}

	/**
	 * Take the current state as a sample.
	 */
	protected void takeSample() {
		if (this.samples.isEmpty()) {

			// initialize list of samples per sentence
			for (int[] sent : this.state) {
				List<IntCounter> sentence = new ArrayList<IntCounter>();
				for (@SuppressWarnings("unused")
				int link : sent) {
					sentence.add(new IntCounter());
				}
				this.samples.add(sentence);
			}
		}

		// add samples per link
		for (int sent = 0; sent < this.state.size(); sent++) {
			int[] sentence = this.state.get(sent);
			List<IntCounter> sentenceSamples = this.samples.get(sent);
			for (int link = 0; link < sentence.length; link++) {
				sentenceSamples.get(link).put(sentence[link], 1.0);
			}
		}
	}

	/**
	 * Train an IBM1 model and use its Viterbi alignment in order to initialize the state of the Markov chain.
	 * 
	 * @param corpus
	 *            A numeric representation of the corpus stored in memory
	 * @throws FileNotFoundException
	 *             If path is wrong
	 * @throws IOException
	 */
	protected abstract void assignInitialAlignment(ParallelCorpus corpus) throws IOException;

	/**
	 * Re-sample alignment links for French words. The sampler that has been specified will be used.
	 * 
	 * @param corpus
	 *            A numeric representation of the corpus stored in memory
	 * @param iterations
	 *            The number of iterations that the sampler will perform
	 * @param burnIn
	 *            The number of initial iterations for which no samples will be taken
	 * @param lag
	 *            The lag between individual samples
	 * @throws FileNotFoundException
	 *             If the path to the file is incorrect
	 * @throws IOException
	 */
	protected abstract void sample(ParallelCorpus corpus, int iterations, int burnIn, int lag) throws IOException;
}
