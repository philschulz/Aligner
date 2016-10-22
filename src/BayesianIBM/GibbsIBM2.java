package BayesianIBM;

import ibmModels.IBM1;
import ibmModels.VBIBM1;
import io.TranslationTableUtils;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

import java.io.IOException;
import java.util.Map;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import sampling.MultivariateLikelihood;
import sampling.MultivariateSliceSampler;
import alignmentUtils.DistortionModel;
import alignmentUtils.ParallelCorpus;
import collections.DoubleCounter;
import collections.IntCounter;
import collections.machineTranslation.LogTranslationTable;

public class GibbsIBM2 extends GibbsIBM1 implements DistortionModel {

	protected Int2DoubleOpenHashMap distortionPrior;
	protected int maxDistortion;
	protected IntCounter distortionCounter;
	protected final int NULL_JUMP = Integer.MIN_VALUE;

	protected GammaDistribution distortionHyperprior;
	protected MultivariateSliceSampler<Integer> distortionHyperSampler;

	/**
	 * Construct a GibbsIBM2 model.
	 * 
	 * @param maxDistortion
	 *            The maximum distortion that the model allows for. All distortions that are greater will be mapped to this limit.
	 */
	public GibbsIBM2(double distortionPrior) {
		super();
		this.distortionCounter = new IntCounter();
		this.maxDistortion = 0;
		setDistortionPrior(distortionPrior);

		// TODO make this adjustable
		this.distortionHyperprior = new GammaDistribution(1, 1);
	}

	/**
	 * Construct a GibbsIBM2 model with the maximum distortion limit set to 5.
	 */
	public GibbsIBM2() {
		this(1);
	}

	/**
	 * Set the Dirichlet parameter for the distortion model.
	 * 
	 * @param distortionPrior
	 *            The Dirichlet parameter
	 * @throws IllegalArgumentException
	 *             if the distortion prior is not positive
	 */
	public void setDistortionPrior(double distortionPrior) throws IllegalArgumentException {
		if (distortionPrior <= 0) {
			throw new IllegalArgumentException("The distortionPrior (Dirichlet parameter) must be positive.");
		}
		this.distortionPrior = new Int2DoubleOpenHashMap();
		this.distortionPrior.put(NULL_JUMP, distortionPrior);

		for (int dist = -maxDistortion; dist <= maxDistortion; dist++) {
			this.distortionPrior.put(dist, distortionPrior);
		}
	}

	public void setMaxDistortion(int maxDistortion) {
		this.maxDistortion = maxDistortion;
	}

	public int computeDistortion(int sourcePos, int targetPos, int sourceLength, int targetLength) {
		int distortion;
		if (targetPos == 0) {
			distortion = this.NULL_JUMP;
		} else {
			double lengthRatio = ((double) (targetLength - 1)) / sourceLength;
			distortion = targetPos - (int) (Math.floor((sourcePos + 1) * lengthRatio));
		}
		return distortion;
	}

	@Override
	protected void removeInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide) {
		super.removeInformation(englishPos, englishSide, frenchPos, frenchSide);
		int distortion = computeDistortion(frenchPos, englishPos, frenchSide.length, englishSide.length);
		this.distortionCounter.subtract(distortion, 1.0);
	}

	@Override
	protected void updateInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide, int[] alignmentVector) {
		super.updateInformation(englishPos, englishSide, frenchPos, frenchSide, alignmentVector);
		int distortion = computeDistortion(frenchPos, englishPos, frenchSide.length, englishSide.length);
		this.distortionCounter.put(distortion, 1.0);
	}

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
			// -2 discounts the NULL position which has a separate distortion value
			this.maxDistortion = englishSide.length - 2 > this.maxDistortion ? englishSide.length - 2 : this.maxDistortion;

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
		setDistortionPrior(this.distortionPrior.get(NULL_JUMP));
	}

	@Override
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

	@Override
	public double scoreDistortion(int sourcePos, int targetPos, int sourceLength, int targetLength) {
		int jump = this.computeDistortion(sourcePos, targetPos, sourceLength, targetLength);
		// System.out.printf("Distortion = %d, DistortionScore = %f, DistortionPrior = %f%n", jump, this.distortionCounter.get(jump),
		// this.distortionPrior.get(jump));
		return this.distortionCounter.get(jump) + this.distortionPrior.get(jump);
	}

	// ////////////////////////////////// Hyperparameter Inference
	// ////////////////////////////////// /////////////////////////////////////////////

	@Override
	protected void sliceSampleHyperparameters(int iterations) {
		//super.sliceSampleHyperparameters(iterations);

		if (this.distortionHyperSampler == null) {
			this.distortionHyperSampler = new MultivariateSliceSampler<Integer>(this.distortionHyperprior, new DistortionLikelihood(
					this.distortionCounter));
		} else {
			this.distortionHyperSampler.setLikelihood(new DistortionLikelihood(distortionCounter));
		}

		this.distortionHyperSampler.sample(this.distortionPrior, iterations, 0, 0);
		for (Map.Entry<Integer, Double> entry : distortionHyperSampler.getState().entrySet()) {
			this.distortionPrior.put(entry.getKey(), entry.getValue());
		}
	}

	@Override
	public double computeLogLikelihood() {
		DistortionLikelihood dl = new DistortionLikelihood(this.distortionCounter);
		double result = dl.compute(this.distortionPrior) + super.computeLogLikelihood();
		if (this.hyper) {
			double priorScore = 0;
			for (double value : this.distortionPrior.values()) {
				priorScore += this.distortionHyperprior.logDensity(value);
			}
			return result + priorScore;
		} else {
			return result;
		}
	}

	private class DistortionLikelihood implements MultivariateLikelihood<Integer> {
		IntCounter distortionCounter;
		Map<Integer, Double> parameterVector;
		double parameterTotal;
		double parameterProduct;
		double likelihoodTotal;
		double likelihoodProduct;
		boolean init;

		public DistortionLikelihood(IntCounter distortionCounter) {
			this.distortionCounter = distortionCounter;
			this.init = false;
		}

		@Override
		public double compute(Map<Integer, Double> parameterVector) throws IllegalArgumentException {
			if (!(this.distortionCounter.size() <= parameterVector.size())) {
				System.err.println("Distortion%n" + this.distortionCounter.toString());
				System.err.println("Parameter%n" + parameterVector);
				throw new IllegalArgumentException("Length of parameter vector does not match number of parameters in likelihood.");
			}

			for (Map.Entry<Integer, Double> entry : this.distortionCounter.entrySet()) {
				Integer key = entry.getKey();
				Double count = entry.getValue();

				Double paramValue = parameterVector.get(key);
				parameterTotal += paramValue;
				parameterProduct += Gamma.logGamma(paramValue);
				likelihoodTotal += paramValue + count;
				likelihoodProduct += Gamma.logGamma(paramValue + count);
			}

			this.init = true;
			this.parameterVector = parameterVector;
			return Gamma.logGamma(parameterTotal) - Gamma.logGamma(likelihoodTotal) + likelihoodProduct - parameterProduct;
		}

		@Override
		public double computeAt(Integer dimension, double value) {
			if (!init) {
				System.out.println("Error in likelihood computation because not initialised");
			}

			double paramSum = this.parameterTotal + value - this.parameterVector.get(dimension);
			double paramProd = this.parameterProduct + Gamma.logGamma(value) - Gamma.logGamma(this.parameterVector.get(dimension));
			double likeSum = this.likelihoodTotal + value - this.parameterVector.get(dimension);
			double likeProd = this.likelihoodProduct + Gamma.logGamma(value + this.distortionCounter.get(dimension))
					- Gamma.logGamma(this.parameterVector.get(dimension) + this.distortionCounter.get(dimension));

			// System.out.printf("ParamTotal = %f, ParamProd = %f, likeTotal =
			// %f, likeProd = %f%n", this.parameterTotal,
			// this.parameterProduct, this.likelihoodTotal,
			// this.likelihoodProduct);

			return Gamma.logGamma(paramSum) - paramProd + likeProd - Gamma.logGamma(likeSum);
		}

		@Override
		public void updateAt(Integer dimension, double value) {
			this.parameterTotal += value - this.parameterVector.get(dimension);
			this.parameterProduct += Gamma.logGamma(value) - Gamma.logGamma(this.parameterVector.get(dimension));
			this.likelihoodTotal += value - this.parameterVector.get(dimension);
			this.likelihoodProduct += Gamma.logGamma(value + this.distortionCounter.get(dimension))
					- Gamma.logGamma(this.parameterVector.get(dimension) + this.distortionCounter.get(dimension));

			this.parameterVector.put(dimension, value);
		}

		@Override
		public boolean hasObservation(Integer dimension) {
			return this.distortionCounter.containsKey(dimension);
		}
	}
}
