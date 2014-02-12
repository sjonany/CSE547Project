package learning.libsvm;

import java.io.PrintWriter;

import util.SvmlightUtil;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;

/**
 * Driver for creating svmlight models trained on given dataset. Reference code
 * = http://www.mpi-inf.mpg.de/~mtb/svmlight/JNI_SVMLight_Test.java
 */
public class SvmlightTrainingDriver {
	/** Names of the files that will be outputted in dir */
	private static final String MAPPING_FILENAME_TEMPLATE = "%s/mapping.txt";
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";

	public static void main(String[] args) throws Exception {
		if (args.length != 2) {
			System.err
			    .println("Usage: <path to train> <output dir - must be a folder, will output all the models + mapping>");
			return;
		}

		String trainFile = args[0];
		String outputDir = args[1];
		if (outputDir.endsWith("/")) {
			outputDir = outputDir.substring(0, outputDir.length() - 1);
		}

		VerbObjectStatComputer stats = new VerbObjectStatComputer();
		System.out.println("Precomputing stats from train file...");
		stats.load(trainFile);
		System.out.println("Finished precomputation.");

	  FeatureExtractor featureExtractor = new VerbCooccurrenceFeatureExtractor(stats);
		
		// Write the verb to id mapping
		PrintWriter mappingWriter = new PrintWriter(String.format(
		    MAPPING_FILENAME_TEMPLATE, outputDir));
		for (int verbId = 1; verbId <= stats.getCountDistinctVerb(); verbId++) {
			mappingWriter.println(verbId + "\t" + stats.mapIdToVerb(verbId));
		}
		mappingWriter.close();

		// For each verb, we train a single model, then write the model to disk
		// Each such iteration requires going through the entire dataset
		for (int verbId = 1; verbId <= stats.getCountDistinctVerb(); verbId++) {
			long curtime = System.currentTimeMillis();
			LabeledFeatureVector[] trainSet = SvmlightUtil.filterDatasetToVerb(trainFile, stats.mapIdToVerb(verbId), featureExtractor);
			
			TrainingParameters trainParam = new TrainingParameters();
			// Switch on some debugging output
			// http://infolab.stanford.edu/~theobald/svmlight/doc/jnisvmlight/LearnParam.html
			trainParam.getLearningParameters().verbosity = 1;
			// TODO: cross-validation for these params
			trainParam.getLearningParameters().svm_costratio = 2.0;

			SVMLightModel model = new SVMLightInterface().trainModel(trainSet, trainParam);
			model.writeModelToFile(String.format(MODEL_FILENAME_TEMPLATE, outputDir,
			    verbId));
			long elapsedTime = System.currentTimeMillis() - curtime;
			System.out.printf("%d out of %d models trained. Takes %d ms\n", verbId,
			    stats.getCountDistinctVerb(), elapsedTime);
		}
	}
}
