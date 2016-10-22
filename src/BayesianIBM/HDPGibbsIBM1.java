package BayesianIBM;

import ibmModels.IBM1;
import io.ParallelReader;
import io.SNTReader;
import it.unimi.dsi.fastutil.doubles.DoubleArrayList;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntList;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import math.StirlingNumbers;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.special.Gamma;

import base.BayesAlign;
import distributions.Multinomial;
import distributions.ProbabilityException;
import alignmentUtils.ParallelCorpus;
import sampling.HyperparameterSliceSampler;
import sampling.Likelihood;
import stochasticProcesses.DirichletProcess;
import stochasticProcesses.HDPChild;
import collections.CountTable;
import collections.Counter;
import collections.IntCountTable;
import collections.IntCounter;

public class HDPGibbsIBM1 extends BayesAlign {

	protected HDPBase root;
	// maps English words to their associated DPs
	protected Int2ObjectOpenHashMap<Child> childProcesses;

	// stores denominators in rows and enumerators in cols
	// TODO Change to DoubleCountTable
	private CountTable<Double> countRatios;
	private Int2ObjectOpenHashMap<int[]> positionsPerSentenceLength;

	private StirlingNumbers stirling;
	protected HyperparameterSliceSampler hyperSampler;
	protected double childConcentration;
	private double rootConcentration;

	public HDPGibbsIBM1(double alpha, Multinomial<Integer> baseDistribution) {
		this.root = new HDPBase(alpha, baseDistribution);
		this.childProcesses = new Int2ObjectOpenHashMap<Child>();
		this.childConcentration = alpha;
		this.rootConcentration = alpha;
		this.countRatios = new CountTable<Double>();
		this.positionsPerSentenceLength = new Int2ObjectOpenHashMap<int[]>();
		this.aux = false;
		this.randomGenerator = new MersenneTwister();
		this.state = new ArrayList<int[]>();
		this.samples = new ArrayList<List<IntCounter>>();
		this.stirling = new StirlingNumbers();
		this.translationGammaA = 1;
		this.translationGammaB = 1;
		this.hyperSampler = new HyperparameterSliceSampler(new GammaDistribution(this.translationGammaA, this.translationGammaB));
	}

	protected void initializeHDP(ParallelCorpus corpus) {
		for (int[][] pair : corpus) {

			int[] targetSide = pair[1];

			for (int target : targetSide) {
				if (!childProcesses.containsKey(target)) {
					childProcesses.put(target, new Child(this.childConcentration, this.root));
				}
			}
		}
	}

	@Override
	protected void assignInitialAlignment(ParallelCorpus corpus) throws IOException {
		initializeHDP(corpus);

		IBM1 baseModel = IBM1.createModel(corpus, 5);

		for (int[][] pair : corpus) {
			int[] sourceSide = pair[0];
			int[] targetSide = pair[1];
			int[] alignmentVector = new int[sourceSide.length];

			for (int sourcePos = 0; sourcePos < sourceSide.length; sourcePos++) {
				int source = sourceSide[sourcePos];
				double highest_score = Double.NEGATIVE_INFINITY;
				int bestLink = -1;

				for (int targetPos = 0; targetPos < targetSide.length; targetPos++) {
					int target = targetSide[targetPos];
					double score;

					if ((score = baseModel.scoreTranslationPair(source, target)) > highest_score) {
						highest_score = score;
						bestLink = targetPos;
					}
				}
				alignmentVector[sourcePos] = bestLink;
				this.childProcesses.get(targetSide[bestLink]).addObservation(source);
			}
			getState().add(alignmentVector);
		}
	}

	protected void removeInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide) {
		int target = englishSide[englishPos];
		int source = frenchSide[frenchPos];

		this.childProcesses.get(target).removeObservation(source);
	}

	protected void updateInformation(int englishPos, int[] englishSide, int frenchPos, int[] frenchSide, int[] alignmentVector) {
		int target = englishSide[englishPos];
		int source = frenchSide[frenchPos];

		this.childProcesses.get(target).addObservation(source);
		alignmentVector[frenchPos] = englishPos;
	}

	/**
	 * Get the ratio enumerator/denominator
	 * 
	 * @param enumerator
	 *            The enumerator of the ratio
	 * @param denominator
	 *            The denominator of the ratio
	 * @return enumerator/denominator
	 */
	protected double countRatio(double enumerator, double denominator) {
		if (enumerator == 0) {
			return 0;
		}

		double result;
		if ((result = this.countRatios.get(denominator, enumerator)) == 0) {
			result = enumerator / denominator;
			this.countRatios.put(denominator, enumerator, result);
		}

		return result;
	}

	public double scoreTranslationPair(int source, int target) {
		return this.childProcesses.get(target).probability(source);
	}

	protected double[][] getAlignmentProbs(int[] englishSide, int[] competition, int[] frenchSide, int frenchPos) {
		double[] probs = new double[englishSide.length];
		double totalProbs = 0;
		int source = frenchSide[frenchPos];

		for (int i : competition) {
			int target = englishSide[i];
			double prob = scoreTranslationPair(source, target);
			totalProbs += prob;
			probs[i] = prob;
		}

		return new double[][] { probs, new double[] { totalProbs } };
	}

	@Override
	public void align(String pathToSNTFile, String outputFile, String format, int iterations, int burnIn, int lag)
			throws FileNotFoundException, IOException {
		ParallelCorpus corpus = ParallelCorpus.readCorpus(pathToSNTFile);
		this.assignInitialAlignment(corpus);

		this.sample(corpus, iterations, burnIn, lag);

		try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile))) {
			writeAlignments(writer, format);
		}
	}

	protected void gibbsSampleOnce(ParallelCorpus corpus) {
		int sentNum = 0;

		for (int[][] pair : corpus) {
			int[] alignmentVector = this.state.get(sentNum);
			int[] sourceSide = pair[0];
			int[] targetSide = pair[1];

			int[] targetPositions;
			if ((targetPositions = this.positionsPerSentenceLength.get(targetSide.length)) == null) {
				targetPositions = new int[targetSide.length];
				for (int i = 0; i < targetSide.length; i++) {
					targetPositions[i] = i;
				}
				this.positionsPerSentenceLength.put(targetSide.length, targetPositions);
			}

			for (int sourcePos = 0; sourcePos < sourceSide.length; sourcePos++) {
				int currentAligned = alignmentVector[sourcePos];

				removeInformation(currentAligned, targetSide, sourcePos, sourceSide);

				int[] competition = defineCompetition(targetSide, currentAligned, targetPositions);
				double[][] alignmentProbs = getAlignmentProbs(targetSide, competition, sourceSide, sourcePos);
				double[] probs = alignmentProbs[0];

				double threshold = this.randomGenerator.nextDouble() * alignmentProbs[1][0];
				int alignmentPos = 0;
				double total = 0;
				for (int competitor : competition) {
					alignmentPos = competitor;
					total += probs[alignmentPos];

					if (total >= threshold) {
						break;
					}
				}

				this.updateInformation(alignmentPos, targetSide, sourcePos, sourceSide, alignmentVector);
			}
			sentNum++;
		}
	}

	@Override
	protected void sample(ParallelCorpus corpus, int iterations, int burnIn, int lag) {
		double decile = Math.floor(iterations / 10.0);
		int samplesTaken = 0;

		System.out.println("Started sampling at " + new Date());

		for (int iter = 1; iter <= iterations; iter++) {
			if (iter % decile == 0) {
				System.out.println("Iteration " + iter);
			}

			long gibbsTime = System.nanoTime();
			gibbsSampleOnce(corpus);
			gibbsTime = System.nanoTime() - gibbsTime;

			System.out.println("Iteration " + iter);
			System.out.printf("Gibbs iteration took %f secs%n", gibbsTime / 1000000000.0);

			if (this.hyper) {
				long hyperTime = System.nanoTime();
				this.sliceSampleHyperparameters(10);
				hyperTime = System.nanoTime() - hyperTime;
				this.countRatios.clear();
				System.out.printf("Hyper-parameter sampling took %f secs%n", hyperTime / 1000000000.0);
			}

			// take a sample
			if (iter >= burnIn && iter % lag == 0) {
				this.takeSample();
				samplesTaken++;
			}
		}
		System.out.println("Finished sampling at " + new Date());
		System.out.printf("%d samples taken in total.%n", samplesTaken);
	}

	protected class HDPBase implements DirichletProcess<Integer, Integer> {

		private double alpha;
		private Multinomial<Integer> baseDistribution;
		private double totalObservations;
		private LinkedList<Integer> componentGraveyard;
		private int maxComponent;
		private IntCountTable distributions2components;
		private IntCounter probCache;
		private int numComponents;

		public HDPBase(double alpha, Multinomial<Integer> distribution) {
			this.alpha = alpha;
			this.baseDistribution = distribution;
			this.totalObservations = alpha;
			this.distributions2components = new IntCountTable();
			this.maxComponent = 0;
			this.componentGraveyard = new LinkedList<Integer>();
			this.probCache = new IntCounter();
			this.numComponents = 0;
		}

		public int getTotalObservations() {
			return (int) Math.round(totalObservations - this.alpha);
		}

		@Override
		public double probability(Integer observation) {
			double prob = probCache.get(observation);
			if (prob != 0) {
				return prob;
			} else {
				IntCounter row = distributions2components.getRow(observation);
				if (row == null) {
					prob = countRatio(this.alpha, totalObservations) * this.baseDistribution.query(observation);
				} else {
					double evidence = row.getTotal();
					prob = countRatio(evidence, totalObservations) + countRatio(this.alpha, totalObservations)
							* this.baseDistribution.query(observation);
				}
				probCache.put(observation, prob);
				return prob;
			}
		}

		/*
		 * Adds a new component into the list of components by assigning it the first unoccupied integer. This way component ids always run
		 * from 0 to the number of components -1.
		 */
		private int newComponentID() {
			int comp;
			if (!componentGraveyard.isEmpty()) {
				comp = componentGraveyard.pop();
			} else {
				comp = this.maxComponent;
				maxComponent++;
			}

			this.numComponents++;
			return comp;
		}

		@Override
		public Integer drawComponent(Integer observation) {
			IntCounter responsibleComponents = distributions2components.getRow(observation);
			if (responsibleComponents == null) {
				// udpate draw of new component in parentDistribution
				return newComponentID();
			} else {
				double baseProb = this.baseDistribution.query(observation) * this.alpha;
				double threshold = randomGenerator.nextDouble() * (responsibleComponents.getTotal() + baseProb);

				if (baseProb >= threshold) {
					return newComponentID();
				} else {
					int countTreshold = randomGenerator.nextInt((int) responsibleComponents.getTotal());
					int winner = -1;
					for (Map.Entry<Integer, Double> component : responsibleComponents.entrySet()) {
						if ((countTreshold -= component.getValue()) < 0) {
							winner = component.getKey();
						}
					}
					return winner;
				}
			}
		}

		@Override
		public void addObservation(Integer observation) {
			int component = drawComponent(observation);

			distributions2components.put(observation, component, 1.0);
			this.totalObservations += 1;
			probCache.clear();

			if (totalObservations - this.numComponents < -0.00001) {
				System.out.print("In Root Removal");
				System.out.printf("TotalObservations = %f, Components = %f%n", totalObservations, this.numComponents);
				System.exit(0);
			}
		}

		@Override
		public boolean removeObservation(Integer observation) {
			IntCounter responsibleComponents = distributions2components.getRow(observation);
			double total = responsibleComponents.getTotal();
			int prevSize = responsibleComponents.size();
			double threshold = randomGenerator.nextDouble() * total;
			double accumulator = 0;
			int selectedComponent = -1;

			for (Map.Entry<Integer, Double> component : responsibleComponents.entrySet()) {
				if ((accumulator += component.getValue()) >= threshold) {
					selectedComponent = component.getKey();
				}
			}
			distributions2components.subtract(observation, selectedComponent, 1.0);
			boolean eliminate = prevSize != responsibleComponents.size();

			if (eliminate) {
				this.numComponents--;
				this.componentGraveyard.add(selectedComponent);
			}
			totalObservations -= 1;
			probCache.clear();

			if (totalObservations - this.numComponents < -0.00001) {
				System.out.print("In Root Removal");
				System.out.printf("TotalObservations = %f, Components = %f%n", totalObservations, this.numComponents);
				System.exit(0);
			}
			return numComponents == 0;
		}

		@Override
		@Deprecated
		public Integer drawComponent() {
			// TODO Auto-generated method stub
			return null;

		}

		@Override
		public int numberOfComponents() {
			return numComponents;
		}

		@Override
		public void setConcentration(double alpha) throws IllegalArgumentException {
			if (alpha <= 0) {
				throw new IllegalArgumentException("Precision Parameter of Dirichlet Process needs to be positive.");
			}
			this.totalObservations += alpha - this.alpha;
			this.alpha = alpha;
		}
	}

	protected class Child implements HDPChild<Integer, Integer> {

		private double alpha;
		private double totalObservations;

		private int maxComponent;
		private DirichletProcess<Integer, Integer> parent;
		public IntCountTable distributions2components;
		private LinkedList<Integer> componentGraveyard;
		private int numComponents;

		public Child(double alpha, DirichletProcess<Integer, Integer> parent) {
			this.alpha = alpha;
			this.parent = parent;
			this.maxComponent = 0;
			// totalObservations is only used in denominator anyway, thus we can pre-add alpha
			this.totalObservations = alpha;
			this.distributions2components = new IntCountTable();
			this.componentGraveyard = new LinkedList<Integer>();
			this.numComponents = 0;
		}

		public int getTotalObservations() {
			return (int) Math.round(totalObservations - this.alpha);
		}

		/*
		 * Adds a new component into the list of components by assigning it the first unoccupied integer. This way component ids always run
		 * from 0 to the number of components -1.
		 */
		private int newComponentID() {
			int comp;
			if (!this.componentGraveyard.isEmpty()) {
				comp = this.componentGraveyard.pop();
			} else {
				comp = this.maxComponent;
				this.maxComponent++;
			}

			this.numComponents++;
			return comp;
		}

		private int getComponentFromParent(int observation) {
			parent.addObservation(observation);
			return newComponentID();
		}

		@Override
		public Integer drawComponent(Integer observation) {
			IntCounter responsibleComponents = distributions2components.getRow(observation);
			if (responsibleComponents == null) {
				// udpate draw of new component in parentDistribution
				return getComponentFromParent(observation);
			} else {
				double newProb = parent.probability(observation) * this.alpha;

				// sample space is restricted to the responsible components (i.e. components with likelihood > 0)
				double threshold = randomGenerator.nextDouble() * (responsibleComponents.getTotal() + newProb);
				if (newProb >= threshold) {
					// udpate draw of new component in parentDistribution
					return getComponentFromParent(observation);
				} else {
					int winner = -1;
					int countThreshold = randomGenerator.nextInt((int) responsibleComponents.getTotal());
					for (Map.Entry<Integer, Double> component : responsibleComponents.entrySet()) {
						if ((countThreshold -= component.getValue()) < 0) {
							winner = component.getKey();
							break;
						}
					}
					return winner;
				}
			}
		}

		@Override
		public DirichletProcess<Integer, Integer> getParent() {
			return parent;
		}

		@Override
		public double probability(Integer observation) {
			IntCounter row = distributions2components.getRow(observation);

			double parentProb = getParent().probability(observation);

			if (row == null) {
				return countRatio(this.alpha, totalObservations) * parentProb;
			} else {
				double evidence = row.getTotal();
				return countRatio(evidence, totalObservations) + countRatio(this.alpha, totalObservations) * parentProb;
			}
		}

		@Override
		public void addObservation(Integer observation) {
			int component = drawComponent(observation);

			distributions2components.put(observation, component, 1.0);
			totalObservations += 1;

			if (totalObservations - this.alpha - this.numComponents < -0.00001) {
				System.out.println("In Child Addition");
				System.out.printf("TotalObservations = %f, Components = %d%n", totalObservations - this.alpha, this.numComponents);
				System.exit(0);
			}
		}

		@Override
		public boolean removeObservation(Integer observation) {
			IntCounter responsibleComponents = distributions2components.getRow(observation);
			double total = responsibleComponents.getTotal();
			int prevSize = responsibleComponents.size();
			double threshold = randomGenerator.nextDouble() * total;
			double accumulator = 0;
			int selectedComponent = -1;

			for (Map.Entry<Integer, Double> component : responsibleComponents.entrySet()) {
				if ((accumulator += component.getValue()) >= threshold) {
					selectedComponent = component.getKey();
				}
			}
			distributions2components.subtract(observation, selectedComponent, 1.0);
			boolean eliminate = prevSize != responsibleComponents.size();

			if (eliminate) {
				this.numComponents--;
				this.componentGraveyard.add(selectedComponent);
				getParent().removeObservation(observation);
			}
			totalObservations -= 1;

			if (totalObservations - this.alpha - this.numComponents < -0.00001) {
				System.out.println("In Child Removal");
				System.out.printf("TotalObservations = %f, Components = %d%n", totalObservations - this.alpha, this.numComponents);
				System.exit(0);
			}
			return numComponents == 0;
		}

		@Override
		@Deprecated
		public Integer drawComponent() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public int numberOfComponents() {
			return this.numComponents;
		}

		@Override
		public void setConcentration(double alpha) throws IllegalArgumentException {
			if (alpha <= 0) {
				throw new IllegalArgumentException("Precision Parameter of Dirichlet Process needs to be positive.");
			}

			this.totalObservations += alpha - this.alpha;
			this.alpha = alpha;
		}
	}

	public static Multinomial<Integer> sourceWordDistribution(String pathToSNTFile) throws FileNotFoundException, IOException,
			ProbabilityException {
		Counter<Integer> wordCounter = new Counter<Integer>();

		System.out.println("Starting to compute source unigram distribution at " + new Date());
		try (ParallelReader reader = new SNTReader(pathToSNTFile)) {
			String[][] nextPair;
			while ((nextPair = reader.getNextPair()) != null) {
				int[] sourceRepresentation = ParallelCorpus.stringToInt(nextPair[0], false);
				for (int word : sourceRepresentation) {
					wordCounter.put(word, 1.0);
				}
			}
		}

		wordCounter.normalize();
		Multinomial<Integer> distribution = Multinomial.toMultinomial(wordCounter.toMap());
		System.out.println("Finished computing source unigram distribution at " + new Date());
		return distribution;
	}

	public static Multinomial<Integer> uniformSourceWordDistribution(String pathToSNTFile) throws FileNotFoundException, IOException,
			ProbabilityException {
		Counter<Integer> wordCounter = new Counter<Integer>();

		System.out.println("Starting to compute source unigram distribution at " + new Date());
		try (ParallelReader reader = new SNTReader(pathToSNTFile)) {
			String[][] nextPair;
			while ((nextPair = reader.getNextPair()) != null) {
				int[] sourceRepresentation = ParallelCorpus.stringToInt(nextPair[0], false);
				for (int word : sourceRepresentation) {
					if (!wordCounter.containsKey(word)) {
						wordCounter.put(word, 1.0);
					}
				}
			}
		}

		wordCounter.normalize();
		Multinomial<Integer> distribution = Multinomial.toMultinomial(wordCounter.toMap());
		System.out.println("Finished computing source unigram distribution at " + new Date());
		return distribution;
	}

	// ////////////////////////// Hyperparameter Inference //////////////////////////////////

	protected void sliceSampleHyperparameters(int iterations) {
		this.sampleChildHyperparameters(iterations);
		this.sampleRootHyperparameters(iterations);
	}

	protected void sampleChildHyperparameters(int iterations) {
		this.hyperSampler.setLikelihood(new ChildConcentrationLikelihood(this.childProcesses.values()));
		this.hyperSampler.sample(this.childConcentration, iterations, 0, 0);
		this.childConcentration = hyperSampler.getState();

		for (Child child : this.childProcesses.values()) {
			child.setConcentration(this.childConcentration);
		}
		System.out.printf("New Child concentration = %f%n", this.childConcentration);
	}

	protected void sampleRootHyperparameters(int iterations) {
		this.hyperSampler.setLikelihood(new RootConcentrationLikelihood(this.root));
		this.hyperSampler.sample(this.rootConcentration, iterations, 0, 0);
		this.rootConcentration = this.hyperSampler.getState();
		this.root.setConcentration(this.rootConcentration);

		System.out.printf("New Root concentration = %f%n", this.rootConcentration);
	}

	private class ChildConcentrationLikelihood implements Likelihood {

		private IntList numComponents;
		private DoubleList observations;

		public ChildConcentrationLikelihood(Collection<Child> childArrangements) {
			this.numComponents = new IntArrayList();
			this.observations = new DoubleArrayList();

			for (Child child : childArrangements) {
				if (child.numComponents > 0) {
					this.numComponents.add(child.numberOfComponents());
					this.observations.add(child.getTotalObservations());
				}
			}
		}

		@Override
		public double compute(double parameter) {
			double result = 0;
			for (int i = 0; i < this.numComponents.size(); i++) {
				result += Math.log(numComponents.get(i) * parameter) + Gamma.logGamma(parameter)
						- Gamma.logGamma(observations.getDouble(i) + parameter);
			}
			return result;
		}
	}

	private class RootConcentrationLikelihood implements Likelihood {
		HDPBase root;

		public RootConcentrationLikelihood(HDPBase root) {
			this.root = root;
		}

		@Override
		public double compute(double parameter) {
			return Math.log(root.numberOfComponents() * parameter) + Gamma.logGamma(parameter)
					- Gamma.logGamma(root.getTotalObservations() + parameter);
		}
	}
}
