package learning.libsvm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.net.URL;

import jnisvmlight.FeatureVector;
import jnisvmlight.SVMLightModel;
import util.StatUtil.ClassificationPerformance;

/**
 * Driver for running trained lightsvm models against a test set Reference code
 * = http://www.mpi-inf.mpg.de/~mtb/svmlight/JNI_SVMLight_Test.java
 */
public class SvmlightTestingDriver {
	/** Names of the files that will be outputted in dir */
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";

	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			System.err
			    .println("Usage: <path to train> <path to test> <models dir - must be a folder, here all the models + mapping reside>");
			return;
		}

		// Train is only needed to compute feature stats
		String trainFile = args[0];
		String testFile = args[1];
		String modelDir = args[2];
		if (modelDir.endsWith("/")) {
			modelDir = modelDir.substring(0, modelDir.length() - 1);
		}

		VerbObjectStatComputer stats = new VerbObjectStatComputer();
		System.out.println("Precomputing stats from train file...");
		stats.load(trainFile);
		System.out.println("Finished precomputation.");

	  FeatureExtractor featureExtractor = new VerbCooccurrenceFeatureExtractor(stats);
		
		// Load all the models
		System.out.println("Loading " + stats.getCountDistinctVerb() + " models...");
		SVMLightModel[] models = new SVMLightModel[stats.getCountDistinctVerb() + 1];
		for (int verbId = 1; verbId <= stats.getCountDistinctVerb(); verbId++) {
			models[verbId] = SVMLightModel.readSVMLightModelFromURL(new URL(String
			    .format("file:" + MODEL_FILENAME_TEMPLATE, modelDir, verbId)));
			if (verbId % 100 == 0)
				System.out.printf("Loaded %d out of %d\n", verbId,
				    stats.getCountDistinctVerb());
		}
		System.out.println("Finished loading " + stats.getCountDistinctVerb()
		    + " models...");

		BufferedReader in = new BufferedReader(new FileReader(testFile));
		String line = in.readLine();
		int lineCount = 0;
		ClassificationPerformance globalResult = new ClassificationPerformance();
		
		while (line != null) {
			String[] toks = line.split("\t");
			String verb = toks[0];
			String obj = toks[1];
			boolean isPositive = Integer.parseInt(toks[3]) == 1;
			FeatureVector featVec = featureExtractor.convertDataPointToFeatureVector(verb, obj, isPositive);
			
			// if haven't seen the verb, just say no.
			double prediction = FeatureExtractor.NEGATIVE_CLASS;
			int verbId = stats.mapVerbToId(verb);
			if (verbId >= 1) {
				prediction = models[stats.mapVerbToId(verb)].classify(featVec);
			}

			if (isPositive) {
				if (prediction > 0) {
					globalResult.tp++;
				} else {
					globalResult.fn++;
				}
			} else {
				if (prediction > 0) {
					globalResult.fp++;
				} else {
					globalResult.tn++;
				}
			}

			line = in.readLine();
			lineCount++;
			if (lineCount % 100 == 0) {
				System.out.printf("Processed %d, acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n", lineCount,
				    globalResult.tp + globalResult.tn, globalResult.getDatasetSize(), globalResult.getAccuracy(), globalResult.getPrecision(),
				    globalResult.getRecall(), globalResult.getFscore());
			}
		}
		in.close();
		System.out.printf("Processed %d, acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n", lineCount,
		    globalResult.tp + globalResult.tn, globalResult.getDatasetSize(), globalResult.getAccuracy(), globalResult.getPrecision(),
		    globalResult.getRecall(), globalResult.getFscore());
		System.out.println("TP = " + globalResult.tp);
		System.out.println("FP = " + globalResult.fp);
		System.out.println("FN = " + globalResult.fn);
		System.out.println("TN = " + globalResult.tn);
	}
}
