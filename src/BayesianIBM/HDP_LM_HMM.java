package BayesianIBM;

import java.io.IOException;
import java.util.Date;

import alignmentUtils.ParallelCorpus;
import collections.machineTranslation.LogTranslationTable;
import ibmModels.IBM1;
import io.TranslationTableUtils;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import stochasticProcesses.IntDP;
import stochasticProcesses.IntHDPChild;

public class HDP_LM_HMM extends CollocationHMM {

	private IntDP lmHDPRoot;
	private Int2ObjectOpenHashMap<IntHDPChild> hdpLmChildren;
	private double childConcentration;
	private double rootConcentration;

	public HDP_LM_HMM(double translationPrior, double transitionPrior) {
		super(translationPrior, transitionPrior, 1);
		this.collocationTable = null;
		this.rootConcentration = 1;
		this.lmHDPRoot = new IntDP(rootConcentration, this.sizeOfSupport);
		this.hdpLmChildren = new Int2ObjectOpenHashMap<IntHDPChild>();
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
					int sourceCount = (int) Math.round(this.translationCountTable.get(englishSide[alignmentPoint], frenchSide[j]));
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

		this.sizeOfSupport = sourceWordFreqs.size();
		this.lmHDPRoot.setSizeOfSupport(this.sizeOfSupport);

		setTransitionPrior(this.transitionPrior.get(0));

		// initialize indicators
		System.out.println("Starting to initialise collocation indicators at " + new Date());
		for (int iter = 1; iter <= 200; iter++) {
			sampleCollocationIndicatorsOnce(corpus);
		}
		System.out.println("Finished initialising collocation indicators at " + new Date());
	}

	@Override
	protected void removeCollocationInformation(int prevWord, int currentWord) {
		int count = (int) Math.round(this.continuations.get(prevWord));
		if (count != 0) {
			this.continuationCounts.subtract(count, 1.0);
			if (count != 1) {
				this.continuationCounts.put(count - 1, 1.0);
			}
		}

		this.continuations.subtract(prevWord, 1.0);
		
		IntHDPChild context;
		if ((context = hdpLmChildren.get(prevWord)) != null) {
			context.removeObservation(currentWord);
		} else {
			System.out.println("Error when trying to remove observation " + currentWord + " from component " + prevWord);
		}
	}

	@Override
	protected void updateCollocationInformation(int prevWord, int currentWord) {
		// adjust conditional counts
		int count = (int) Math.round(this.continuations.get(prevWord));
		if (this.continuationCounts.containsKey(count)) {
			this.continuationCounts.subtract(count, 1.0);
		}
		this.continuationCounts.put(count + 1, 1.0);

		this.continuations.put(prevWord, 1.0);

		IntHDPChild context;
		if ((context = hdpLmChildren.get(prevWord)) == null) {
			context = new IntHDPChild(this.childConcentration, this.lmHDPRoot);
			context.addObservation(currentWord);
			hdpLmChildren.put(prevWord, context);
		} else {
			context.addObservation(currentWord);
		}
	}
	
	private double collocationProb(int prevWord, int currentWord) {
		IntHDPChild context;
		double prob;
		
		if ((context = this.hdpLmChildren.get(prevWord)) == null) {
			prob = this.lmHDPRoot.probability(currentWord);
		} else {
			prob = context.probability(currentWord);
		}
		
		return prob;
	}
	
	@Override
	protected boolean sampleCollocationIndicator(int prevLink, int currentLink, int[] englishSide, int prevFrenchPos,
			int currentFrenchPos, int[] frenchSide) {
		int prevWord = prevFrenchPos == -1 ? this.SOURCE_START_SYMBOL : frenchSide[prevFrenchPos];
		int currentWord = frenchSide[currentFrenchPos];
		int alignedWord = englishSide[currentLink];

		double continuationProb = this.continuations.get(prevWord) + this.collocationPrior;
		double collocationProb = collocationProb(prevWord, currentWord);
		// double distortionProb = this.scoreTransition(prevLink, currentLink);
		double alignProb = (this.translationCountTable.get(alignedWord, currentWord) + this.childConcentration)
				/ (this.englishTotals.get(alignedWord) + this.translationPriorTotal);
		double alignmentProb = alignProb;

		double threshold = this.randomGenerator.nextDouble() * (sourceWordFreqs.get(prevWord) + betaTotal)
				* (alignmentProb + collocationProb);
		return continuationProb * collocationProb > threshold;
	}
}
