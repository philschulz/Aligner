package BayesianIBM;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.GammaDistribution;

import sampling.AsymmetricDirichletLikelihood;
import sampling.MultivariateLikelihood;
import sampling.MultivariateSliceSampler;
import alignmentUtils.ParallelCorpus;
import collections.DoubleCounter;
import collections.IntCounter;
import collections.machineTranslation.TranslationTable;

public class BayesHMM extends GibbsIBM1 {

	protected IntCounter transitionCounts;
	protected IntCounter transitionPrior;
	protected int maxJump;
	// jumps from the start state are modeled as forward jumps
	protected final int START_STATE = Integer.MIN_VALUE + 1;
	// the end state is only used as an indicator for there being no forward jump
	protected final int END_STATE = Integer.MIN_VALUE + 2;
	protected final int JUMP_TO_NULL = Integer.MAX_VALUE;
	protected final int JUMP_FROM_NULL = Integer.MAX_VALUE - 1;
	private MultivariateSliceSampler<Integer> distortionHyperSampler;

	public BayesHMM(double translationPrior, double transitionPrior) {
		super(translationPrior);
		transitionCounts = new IntCounter();
		this.maxJump = 0;
		setTransitionPrior(transitionPrior);
		this.distortionHyperSampler = new MultivariateSliceSampler<Integer>(new GammaDistribution(1, 10), null);
	}

	public void setTransitionPrior(double prior) {
		IntCounter transitionPrior = new IntCounter();
		transitionPrior.put(JUMP_TO_NULL, prior);
		transitionPrior.put(JUMP_FROM_NULL, prior);
		for (int i = -maxJump; i <= maxJump; i++) {
			transitionPrior.put(i, prior);
		}
		this.transitionPrior = transitionPrior;
	}

	protected void removeInformation(int prevLink, int currentLink, int nextLink, int[] englishSide, int frenchPos, int[] frenchSide) {
		int englishWord = englishSide[currentLink];
		int frenchWord = frenchSide[frenchPos];
		int jumpToThisLink = computeJump(prevLink, currentLink);
		this.transitionCounts.subtract(jumpToThisLink, 1.0);
		if (nextLink != this.END_STATE) {
			this.transitionCounts.subtract(computeJump(currentLink, nextLink), 1.0);
		}

		int sourceCount = (int) Math.round(this.translationCountTable.get(englishWord, frenchWord));
		this.conditionalSourceCounts.subtract(sourceCount, 1.0);
		if (sourceCount != 1) {
			this.conditionalSourceCounts.put(sourceCount - 1, 1.0);
		}
		
		this.translationCountTable.subtract(englishWord, frenchWord, 1.0);
		this.englishTotals.subtract(englishWord, 1.0);
	}

	protected void updateInformation(int prevLink, int currentLink, int nextLink, int[] englishSide, int frenchPos, int[] frenchSide) {
		int englishWord = englishSide[currentLink];
		int frenchWord = frenchSide[frenchPos];
		int jumpToThisLink = computeJump(prevLink, currentLink);
		this.transitionCounts.put(jumpToThisLink, 1.0);
		if (nextLink != this.END_STATE) {
			this.transitionCounts.put(computeJump(currentLink, nextLink), 1.0);
		}

		int sourceCount = (int) Math.round(this.translationCountTable.get(englishWord, frenchWord));
		if (this.conditionalSourceCounts.containsKey(sourceCount)) {
			this.conditionalSourceCounts.subtract(sourceCount, 1.0);
		}
		this.conditionalSourceCounts.put(sourceCount + 1, 1.0);
		
		this.translationCountTable.put(englishWord, frenchWord, 1.0);
		this.englishTotals.put(englishWord, 1.0);
	}

	protected int computeJump(int prevPos, int currentPos) {
		if (currentPos == 0) {
			return JUMP_TO_NULL;
		} else if (prevPos == 0) {
			return JUMP_FROM_NULL;
		} else if (prevPos == this.START_STATE) {
			return currentPos;
		} else {
			// forward jumps are positive, backward jumps are negative
			return currentPos - prevPos;
		}
	}

	@Override
	protected void assignInitialAlignment(ParallelCorpus corpus) throws IOException {
		super.assignInitialAlignment(corpus);

		int sent = 0;
		for (int[] alignmentVector : this.state) {
			int jump = computeJump(1, corpus.get(sent)[1].length - 1);
			this.maxJump = jump > this.maxJump ? jump : this.maxJump;
			
			int prevLink = this.START_STATE;
			// for the last state, we don't need to compute a forward move -> it is never a prevLink
			for (int j = 0; j < alignmentVector.length; j++) {
				int currentLink = alignmentVector[j];
				this.transitionCounts.put(computeJump(prevLink, currentLink), 1.0);
				prevLink = currentLink;
			}
			sent++;
		}
		setTransitionPrior(this.transitionPrior.get(0));
	}

	protected double scoreTransition(int prevLink, int currentLink, int nextLink) {
		return scoreTransition(prevLink, currentLink) * scoreTransition(currentLink, nextLink);
	}

	protected double scoreTransition(int prevLink, int currentLink) {
		if (currentLink == this.END_STATE) {
			return 1;
		} else {
			int jump = computeJump(prevLink, currentLink);
			return this.transitionCounts.get(jump) + this.transitionPrior.get(jump);
		}
	}

	protected double[][] getAlignmentProbs(int prevLink, int nextLink, int[] englishSide, int[] competition, int[] frenchSide,
			int frenchPos, DoubleCounter probabilityCache, double translationPriorTotal) {
		double[] probs = new double[englishSide.length];
		double totalProbs = 0;
		int frenchWord = frenchSide[frenchPos];

		// compute alignment probs
		for (int i : competition) {
			int englishWord = englishSide[i];
			double count = this.translationCountTable.get(englishWord, frenchWord);
			double total = this.englishTotals.get(englishWord);

			double alignmentProb = scoreTranslationPair(englishWord, frenchWord, count, total, translationPriorTotal, probabilityCache);
			// englishSide.length - 1 because the NULL word is not treated as a position
			alignmentProb *= this.scoreTransition(prevLink, i, nextLink);
			probs[i] = alignmentProb;
			totalProbs += alignmentProb;
		}
		return new double[][] { probs, new double[] { totalProbs } };
	}

	@Override
	protected void gibbsSampleOnce(ParallelCorpus corpus, HashMap<Integer, int[]> positionsPerSentenceLength, double translationPriorTotal,
			DoubleCounter probabilityCache, TranslationTable<Integer> baseDistribution) {
		int sentencePair = 0;
		for (int[][] nextPair : corpus) {
			int[] frenchSide = nextPair[0];
			int[] englishSide = nextPair[1];

			int[] englishPositions = getEnglishPositions(englishSide.length, positionsPerSentenceLength);
			int[] alignment = this.state.get(sentencePair);
			int prevLink = this.START_STATE;

			for (int j = 0; j < alignment.length; j++) {
				int currentLink = alignment[j];
				int nextLink = j == alignment.length - 1 ? this.END_STATE : alignment[j + 1];
				// System.out.printf("Source Pos = %d, prevLink = %d, currentLink = %d, nextLink = %d%n", j, prevLink, currentLink,
				// nextLink);

				// define the competition for the current link
				int[] competition = defineCompetition(englishSide, currentLink, englishPositions);

				// remove all information obtained from this alignment link
				removeInformation(prevLink, currentLink, nextLink, englishSide, j, frenchSide);

				double[][] alignProbs = getAlignmentProbs(prevLink, nextLink, englishSide, competition, frenchSide, j, probabilityCache,
						translationPriorTotal);
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
						currentLink = alignmentPosition;
						break;
					}
				}

				// update with information obtained from new alignment position
				this.updateInformation(prevLink, currentLink, nextLink, englishSide, j, frenchSide);
				alignment[j] = currentLink;
				prevLink = currentLink;
			}
			sentencePair++;
		}
	}

	@Override
	public double computeLogLikelihood() {
		double likelihood = super.computeLogLikelihood();
		MultivariateLikelihood<Integer> distortionLikelihood = new AsymmetricDirichletLikelihood<Integer>(this.transitionCounts.toMap());
		likelihood += distortionLikelihood.compute(this.transitionPrior.toMap());
		return likelihood;
	}

	protected void sliceSampleHyperparameters(int iterations) {
		// super.sliceSampleHyperparameters(iterations);
		this.distortionHyperSampler.setLikelihood(new AsymmetricDirichletLikelihood<Integer>(this.transitionCounts.toMap()));
		this.distortionHyperSampler.sample(this.transitionPrior.toMap(), iterations, 0, 0);

		for (Map.Entry<Integer, Double> entry : this.distortionHyperSampler.getState().entrySet()) {
			this.transitionPrior.put(entry.getKey(), entry.getValue());
		}
	}
	
	@Deprecated
	protected void removeInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide) {
	}
	
	@Deprecated 
	protected void updateInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide) {
	}
}
