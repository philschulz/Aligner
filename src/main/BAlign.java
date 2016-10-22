package main;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import mainUtils.InputChecker;

import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPException;
import com.martiansoftware.jsap.JSAPResult;
import com.martiansoftware.jsap.Switch;
import com.martiansoftware.jsap.UnflaggedOption;

import distributions.Multinomial;
import BayesianIBM.BayesHMM;
import BayesianIBM.CollocationHMM;
import BayesianIBM.CollocationIBM1;
import BayesianIBM.CollocationIBM2;
import BayesianIBM.GibbsIBM1;
import BayesianIBM.GibbsIBM2;
import BayesianIBM.HDPGibbsIBM1;
import BayesianIBM.HDP_LM_HMM;

public class BAlign {

	public static void main(String[] args) throws FileNotFoundException, IOException {

		JSAP commandLineParser = new JSAP();

		List<String> validFormats = Arrays.asList("moses", "talp", "naacl");
		List<String> validModels = Arrays.asList("ibm1", "ibm2", "hdp-ibm1", "collocation-ibm1", "collocation-ibm2",
				"hmm", "collocation-hmm", "hdp-lm-hmm");

		final String usage = "java -jar BAlign.jar";

		try {

			// create and register model option
			FlaggedOption model = new FlaggedOption("model").setStringParser(JSAP.STRING_PARSER).setLongFlag("model")
					.setShortFlag('m').setDefault("ibm1");
			model.setHelp("Specifiy the model to be used. Available models are " + validModels);
			commandLineParser.registerParameter(model);

			// create and register iterations option
			FlaggedOption iterations = new FlaggedOption("iterations").setLongFlag("iterations").setShortFlag('i')
					.setStringParser(JSAP.INTEGER_PARSER).setRequired(true).setDefault("0");
			iterations.setHelp("Specifies the number iteration for which the Gibbs sampler will be running.");
			commandLineParser.registerParameter(iterations);

			// create and register lag option
			FlaggedOption sampleLag = new FlaggedOption("sampleLag").setLongFlag("lag").setShortFlag('l')
					.setStringParser(JSAP.INTEGER_PARSER).setDefault("1");
			sampleLag.setHelp(
					"Speficies the lag between inidividual samples. Sample lag is used to reduce the auto-correlation between the individual samples. If this option "
							+ "is not used, every sample will be stored. This is hugely inefficient in terms of memory and speed. Therefore setting the sample lag manually is recommended.");
			commandLineParser.registerParameter(sampleLag);

			// create and register burn-in option
			FlaggedOption burn = new FlaggedOption("burnIn").setLongFlag("burn-in").setStringParser(JSAP.INTEGER_PARSER)
					.setDefault("0");
			burn.setHelp(
					"Allows for the specification of a burn-in period. Burn in is a heuristic that may or may not help to sample from the true (converged) distribution. "
							+ "Samples will only be taken after the burn-in period has passed. Notice that burn-in always needs to be lower than the total number of iterations.");
			commandLineParser.registerParameter(burn);

			// create and register prior option
			FlaggedOption translationPrior = new FlaggedOption("translationPrior").setLongFlag("translationPrior")
					.setShortFlag('p').setStringParser(JSAP.DOUBLE_PARSER).setRequired(true);
			translationPrior.setHelp(
					"Sets the value or the hyperparameter for the (symmetric) Dirichlet prior. Higher values (> 1) express encourage flatter "
							+ "distributions whereas lower values (< 1) encourage sparser and more concentrated distributions. A hyperparameter with value 1 does not "
							+ "favour any distribution. For word alignment we recommend to use the low values. Note that the hyperparameter needs to be positive.");
			commandLineParser.registerParameter(translationPrior);

			// create and register distortionPrior option
			FlaggedOption distortionPrior = new FlaggedOption("distortionPrior").setStringParser(JSAP.DOUBLE_PARSER)
					.setLongFlag("distortionPrior").setShortFlag('d').setDefault("1");
			distortionPrior.setHelp(
					"Sets the value for the parameter of the symmetric Dirichlet distortion prior. Only takes effect if "
							+ "IBM2 is chosen. Notice that the prior needs to be positive.");
			commandLineParser.registerParameter(distortionPrior);

			// create and register gammaShape option
			FlaggedOption gammaShape = new FlaggedOption("gammaShape").setStringParser(JSAP.DOUBLE_PARSER)
					.setLongFlag("gammaShape").setShortFlag('a').setDefault("0");
			gammaShape.setHelp(
					"Set the shape parameter alpha of the gamma prior for the poisson-ibm2 model. If this model is used, the gamma paramters need to be set. "
							+ "Otherwise, setting them has not effect.");
			commandLineParser.registerParameter(gammaShape);

			// create and register gammaRate option
			FlaggedOption gammaRate = new FlaggedOption("gammaRate").setStringParser(JSAP.DOUBLE_PARSER)
					.setLongFlag("gammaRate").setShortFlag('b').setDefault("0");
			gammaRate.setHelp(
					"Set the rate parameter beta of the gamma prior for the poisson-ibm2 model. If this model is used, the gamma paramters need to be set. "
							+ "Otherwise, setting them has not effect.");
			commandLineParser.registerParameter(gammaRate);

			// create and register format option
			FlaggedOption outputFormat = new FlaggedOption("format").setLongFlag("format").setShortFlag('f')
					.setStringParser(JSAP.STRING_PARSER).setDefault("moses");
			outputFormat.setHelp(String
					.format("Determines the format of the output alignment file. Has to be one of %s. The default is 'Moses'. "
							+ "Arguments are case-insensitive.", validFormats));
			commandLineParser.registerParameter(outputFormat);

			// create and register print option
			FlaggedOption print = new FlaggedOption("print").setLongFlag("print-alignments-during-experiment")
					.setStringParser(JSAP.INTEGER_PARSER).setDefault("0");
			print.setHelp(
					"Set this option to indicate after how many iterations there should be a dump of the currently best alignments. This option is "
							+ "useful when running experiments. If it set to 5, say, alignment will be written do disk every 5 iterations.");
			commandLineParser.registerParameter(print);

			// create and register ibm1Table option
			FlaggedOption table = new FlaggedOption("ibm1Table").setLongFlag("ibm1Table").setDefault("NONE")
					.setStringParser(JSAP.STRING_PARSER);
			table.setHelp(
					"Specify a path to a translation table with this option. The model will then read the table and initialize the IBM1 model with it.");
			commandLineParser.registerParameter(table);

			FlaggedOption lexicalPrior = new FlaggedOption("lexicalPrior").setStringParser(JSAP.DOUBLE_PARSER)
					.setLongFlag("lexical-prior").setDefault("0");
			lexicalPrior.setHelp(
					"Set the lexical emission prior for target language lexical items when using the model working with classes "
							+ "on the sentence level.");
			commandLineParser.registerParameter(lexicalPrior);

			// create and register asymmetric option
			Switch asym = new Switch("asymmetric-prior").setLongFlag("asymmetric-prior");
			asym.setHelp(
					"Uses the translation table learned by the initial IBM1 model to induce an asymmetric prior. The value of the prior variable will then "
							+ "be used as concentration parameter");
			commandLineParser.registerParameter(asym);

			// create and register translation table option
			Switch ttable = new Switch("translation-table").setLongFlag("translation-table").setShortFlag('t');
			ttable.setHelp(
					"Dump the translation table to disk whenever it writing out the alignments during an experiment. If using the class-based model, all "
							+ "translation tables will be dumped. This can take a lot of time, however. I therefore recommend not doing it when using frequent dumps.");
			commandLineParser.registerParameter(ttable);

			// create and register aux option
			Switch auxiliary = new Switch("aux").setLongFlag("aux");
			auxiliary.setHelp(
					"Use an auxiliary variable that uniformly chooses one competitor for the current alignment link. The resampling then takes place between "
							+ "the current link and the competitor. This makes sampling complexity indpendent of sentence length and hence speeds up sampling but may also slow "
							+ "down the mixing of the Markov chain.");
			commandLineParser.registerParameter(auxiliary);

			// create and register likelihood option
			Switch likelihood = new Switch("likelihood").setLongFlag("likelihood");
			likelihood.setHelp(
					"Compute the log-likelihood of the alignment configuration and print it after each iteration. This can give indication for convergence of the Markov chain "
							+ "underlying the Gibbs sampler. In many cases the posterior distribution will only computed up to proportionality and thus the likelihood will be "
							+ "replaced by a quantity that is proportional to it.");
			commandLineParser.registerParameter(likelihood);

			// create and register help option
			Switch help = new Switch("help").setLongFlag("help").setShortFlag('h');
			help.setHelp("Prints this manual.");
			commandLineParser.registerParameter(help);

			// create and register hyper-parameter-inference option
			Switch hyper = new Switch("hyper").setLongFlag("hyper-parameter-inference");
			hyper.setHelp(
					"Triggers hyperparameter inference for all hyperparameters. In this case, the parameters of the corresponding "
							+ "hyperpriors need to be specified.");
			commandLineParser.registerParameter(hyper);

			// create and register align-all option
			Switch alignAll = new Switch("alignAll").setLongFlag("align-all");
			alignAll.setHelp(
					"When using the collocation models, this option will align all source words. Source words that "
							+ "were generated from the language model component are aligned to the target word that the preceding source "
							+ "word is aligned to. If the first word in a source sentence is generated from the language model, it gets aligned "
							+ "to the same position as the first aligned source word in the sentence.");
			commandLineParser.registerParameter(alignAll);

			// create and register hyper-after-sample option
			Switch hyperAfterSample = new Switch("hyperAfterSample").setLongFlag("hyper-after-sample");
			hyperAfterSample
					.setHelp("Activates hyperparameter inference, however, new hyperparameters are only sampled after "
							+ "a sample of the alignment variables has been taken.");
			commandLineParser.registerParameter(hyperAfterSample);

			// create and register input file
			UnflaggedOption file = new UnflaggedOption("file").setStringParser(JSAP.STRING_PARSER).setRequired(true)
					.setGreedy(true);
			file.setHelp("The file(s) containing the parallel corpus to be aligned.");
			commandLineParser.registerParameter(file);

		} catch (JSAPException e) {
			System.err.println(e.getMessage());
			System.exit(-1);
		}

		// extract and check command line input
		JSAPResult commandLine = commandLineParser.parse(args);
		InputChecker.CheckCommandLineInput(commandLine, commandLineParser, usage);

		String model = commandLine.getString("model").toLowerCase();
		String pathToSNTFile = commandLine.getString("file");
		String format = commandLine.getString("format").toLowerCase();
		String ibm1Table = commandLine.getString("ibm1Table");
		int iter = commandLine.getInt("iterations");
		int lag = commandLine.getInt("sampleLag");
		int burnIn = commandLine.getInt("burnIn");
		int experimentPrintOut = commandLine.getInt("print");
		double translationPrior = commandLine.getDouble("translationPrior");
		double distortionPrior = commandLine.getDouble("distortionPrior");
		double lexicalPrior = commandLine.getDouble("lexicalPrior");
		double collocationPriorA = commandLine.getDouble("gammaShape");
		double collocationPriorB = commandLine.getDouble("gammaRate");
		boolean translationTable = commandLine.getBoolean("translation-table");
		boolean aux = commandLine.getBoolean("aux");
		boolean likelihood = commandLine.getBoolean("likelihood");
		boolean hyper = commandLine.getBoolean("hyper");
		boolean alignAll = commandLine.getBoolean("alignAll");
		boolean hyperAfterSample = commandLine.getBoolean("hyperAfterSample");

		// check whether all options are set correctly
		if (!validModels.contains(model)) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.println("Error: the given model is not valid. It has to be one of " + validModels);
			System.exit(0);
		} else if (!validFormats.contains(format)) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.println("Error: The specified format is not valid");
			System.exit(0);
		} else if (iter < 1) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.println("Error: The number of iterations must be positive.");
			System.exit(0);
		} else if (lag < 1 || lag > iter) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.println(
					"Error: The sample lag must be positive and not greater than the total number of iterations.");
			System.exit(-0);
		} else if (burnIn < 0 || burnIn > iter) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.println(
					"Error: The value for burn-in must not be negative and not greater than the total number of iterations.");
			System.exit(0);
		} else if (experimentPrintOut < 0) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.println("Error: The value of print-alignments-during-experiment needs to be positive");
			System.exit(0);
		} else if (model.startsWith("collocation") && lexicalPrior <= 0) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.printf(
					"When using the collocation models a lexical prior greater than 0 needs to be specified (currently %f)%n",
					lexicalPrior);
			System.exit(0);
		} else if (model.startsWith("collocation") && (collocationPriorA <= 0 || collocationPriorB <= 0)) {
			InputChecker.printHelp(commandLineParser, usage);
			System.err.printf(
					"When using the collocation models, positive parameters for the Beta prior on collocation parameters need to be "
							+ " specified (currently they are a = %f and b = %f%n).",
					collocationPriorA, collocationPriorB);
		}

		if (model.equals("ibm1")) {
			GibbsIBM1 aligner = new GibbsIBM1();
			if (likelihood) {
				aligner.printLikelihood();
			}
			aligner.setTranslationPrior(translationPrior);

			System.err.print("Initializing model ");
			if (!ibm1Table.equals("NONE")) {
				System.err.println("with " + ibm1Table + ".");
				aligner.initializeIBM1FromTable(ibm1Table);
			} else {
				System.err.println("by training standard IBM1 for 5 rounds.");
			}
			if (experimentPrintOut > 0) {
				aligner.setExperimentPrintOut(experimentPrintOut);
				aligner.setTTable(translationTable);
			}
			if (hyper) {
				aligner.doHyperparameterInference();
			}
			aligner.setAux(aux);
			aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);

		} else if (model.equals("ibm2")) {
			GibbsIBM2 aligner = new GibbsIBM2(distortionPrior);
			aligner.setTranslationPrior(translationPrior);
			if (likelihood) {
				aligner.printLikelihood();
			}

			System.err.print("Initializing model ");

			if (!ibm1Table.equals("NONE")) {
				System.err.println("with " + ibm1Table + ".");
				aligner.initializeIBM1FromTable(ibm1Table);
			} else {
				System.err.println("by training standard IBM1 for 5 rounds.");
			}
			if (experimentPrintOut > 0) {
				aligner.setExperimentPrintOut(experimentPrintOut);
				aligner.setTTable(translationTable);
			}

			if (hyper) {
				aligner.doHyperparameterInference();
			}
			aligner.setAux(aux);
			aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);
		} else if (model.equals("hdp-ibm1")) {
			Multinomial<Integer> baseDistribution = null;
			try {
				baseDistribution = HDPGibbsIBM1.sourceWordDistribution(pathToSNTFile);
			} catch (Exception e) {
				System.err.println(e);
				System.exit(-1);
			}

			if (model.endsWith("ibm1")) {
				HDPGibbsIBM1 aligner = new HDPGibbsIBM1(translationPrior, baseDistribution);
				// setSampler(aligner, sampler);

				// if (experimentPrintOut > 0) {
				// aligner.setExperimentPrintOut(experimentPrintOut);
				// aligner.setTTable(translationTable);
				// }

				if (hyper) {
					aligner.doHyperparameterInference();
				}
				aligner.setAux(aux);
				aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);
			}
		} else if (model.equals("collocation-ibm1")) {
			CollocationIBM1 aligner = new CollocationIBM1();
			aligner.setAux(aux);
			aligner.setTranslationPrior(translationPrior);
			aligner.setCollocationPrior(collocationPriorA, collocationPriorB);
			aligner.setLMPrior(lexicalPrior);

			System.err.print("Initializing model ");
			System.err.println("by training standard IBM1 for 5 rounds.");

			if (likelihood) {
				aligner.printLikelihood();
			}
			if (hyper) {
				aligner.doHyperparameterInference();
			}
			aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);
		} else if (model.equals("collocation-ibm2")) {
			CollocationIBM2 aligner = new CollocationIBM2(translationPrior, distortionPrior, lexicalPrior);
			aligner.setAux(aux);
			aligner.setCollocationPrior(collocationPriorA, collocationPriorB);

			System.err.print("Initializing model ");
			System.err.println("by training standard IBM1 for 5 rounds.");

			if (hyper) {
				aligner.doHyperparameterInference();
			}
			if (likelihood) {
				aligner.printLikelihood();
			}
			aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);
		} else if (model.equals("collocation-hmm")) {
			CollocationHMM aligner = new CollocationHMM(translationPrior, distortionPrior, lexicalPrior);
			aligner.setAux(aux);
			aligner.setCollocationPrior(collocationPriorA, collocationPriorB);
			aligner.setAlignAll(alignAll);

			System.err.print("Initializing model ");
			System.err.println("by training standard IBM1 for 5 rounds.");

			if (hyper) {
				aligner.doHyperparameterInference();
			}
			if (hyperAfterSample) {
				aligner.hyperAfterSample(true);
			}

			aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);
		} else if (model.equals("hmm")) {
			BayesHMM aligner = new BayesHMM(translationPrior, distortionPrior);
			aligner.setTranslationPrior(translationPrior);
			aligner.setAux(aux);

			if (likelihood) {
				aligner.printLikelihood();
			}
			if (hyper) {
				aligner.doHyperparameterInference();
			}
			aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);
		} else if (model.equals("hdp-lm-hmm")) {
			HDP_LM_HMM aligner = new HDP_LM_HMM(translationPrior, distortionPrior);
			aligner.setAux(aux);
			aligner.setCollocationPrior(collocationPriorA, collocationPriorB);
			aligner.setAlignAll(alignAll);

			aligner.hyperAfterSample(true);
			aligner.align(pathToSNTFile, "alignments", format, iter, burnIn, lag);
		}
	}
}
