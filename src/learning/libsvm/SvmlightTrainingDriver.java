package learning.libsvm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;
import util.StatUtil.ClassificationPerformance;
import util.SvmlightUtil;
import util.WordnetCluster;

/**
 * Driver for creating svmlight models trained on given dataset. Reference code
 * = http://www.mpi-inf.mpg.de/~mtb/svmlight/JNI_SVMLight_Test.java
 */
public class SvmlightTrainingDriver {
	/** Names of the files that will be outputted in dir */
	private static final String MAPPING_FILENAME_TEMPLATE = "%s/mapping.txt";
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";

	public static void main(String[] args) throws Exception {
		if (args.length != 6) {
			System.err
			    .println("Usage: <path to train> <path to validation> <path to stat> <output dir - must be a folder, will output all the models + mapping> " +
			    		"<wordnet path: string> <path to verbs to train>");
			return;
		}

		String trainFile = args[0];
		String validationFile = args[1];
		String statFile = args[2];
		String outputDir = args[3];
		String wordnetPath = args[4];
		String verbToTrainPath = args[5];
		if (outputDir.endsWith("/")) {
			outputDir = outputDir.substring(0, outputDir.length() - 1);
		}

		WordnetCluster wordnet = new WordnetCluster(wordnetPath);
		VerbObjectStatComputer stats = new VerbObjectStatComputer();
		System.out.println("Precomputing stats from train file...");
		stats.load(statFile);
		System.out.println("Finished precomputation.");
		 
	  FeatureExtractor featureExtractor = new VerbCooccurrenceAndSemanticFeatureExtractor(stats, wordnet);
		
		// Write the verb to id mapping
		PrintWriter mappingWriter = new PrintWriter(String.format(
		    MAPPING_FILENAME_TEMPLATE, outputDir));
		for (int verbId = 1; verbId <= stats.getCountDistinctVerb(); verbId++) {
			mappingWriter.println(verbId + "\t" + stats.mapIdToVerb(verbId));
		}
		mappingWriter.close();

		// For each verb, we train a single model, then write the model to disk
		// Each such iteration requires going through the entire dataset
		double[] lambdas = {1, 10, 100, 1000, 10000};
		BufferedReader verbToTrainReader = new BufferedReader(new FileReader(verbToTrainPath));
		String line = verbToTrainReader.readLine();
		int lineCount = 0;
		while(line != null) {
			long curtime = System.currentTimeMillis();
			String verbStr = line.split("\t")[0];
			LabeledFeatureVector[] trainSet = SvmlightUtil.filterDatasetToVerb(trainFile, verbStr, featureExtractor);
	    LabeledFeatureVector[] validationSet = SvmlightUtil.filterDatasetToVerb(validationFile, verbStr, featureExtractor); 
			SVMLightModel[] models = new SVMLightModel[lambdas.length];
			
			// This verb cluster has been discarded from the training set because it has too few training examples. 
			if(trainSet.length == 0 || validationSet.length == 0) { 
				continue;
			}
			int bestLambdaIndex = -1;
			double maxF1 = Double.NEGATIVE_INFINITY;
			// Pick the lambda that works best against the validation set
			for(int lambdaIndex = 0; lambdaIndex < lambdas.length; lambdaIndex++) {
				double lambda = lambdas[lambdaIndex];
				
				TrainingParameters trainParam = new TrainingParameters();
				// Switch on some debugging output
				// http://infolab.stanford.edu/~theobald/svmlight/doc/jnisvmlight/LearnParam.html
				//trainParam.getLearningParameters().verbosity = 1;
				trainParam.getLearningParameters().svm_costratio = 2.0;
				trainParam.getLearningParameters().svm_c = lambda;
			
				SVMLightModel model = new SVMLightInterface().trainModel(trainSet, trainParam);
				models[lambdaIndex] = model;
		    ClassificationPerformance validationResult = SvmlightUtil.testModel(model, validationSet);
				if(bestLambdaIndex == -1) {
					bestLambdaIndex = lambdaIndex;
					maxF1 = validationResult.getFscore();
				} else {
					double fscore = validationResult.getFscore();
					if(!Double.isNaN(fscore) && fscore > maxF1) {
						maxF1 = fscore;
						bestLambdaIndex = lambdaIndex;
					}
				}
				System.out.printf("Verb = %s, Lambda = %.6f, validation f1 = %.6f, prec = %.6f, recall = %.6f, acc = %.6f\n", verbStr,
				    lambdas[lambdaIndex], validationResult.getFscore(), validationResult.getPrecision(), 
				    validationResult.getRecall(), validationResult.getAccuracy());
			}// end lambda
			long elapsedTime = System.currentTimeMillis() - curtime;
			
			int verbId = stats.mapVerbToId(verbStr);
			models[bestLambdaIndex].writeModelToFile(String.format(MODEL_FILENAME_TEMPLATE, outputDir,
			    verbId));
			System.out.printf("Takes %d ms. Best Lambda = %.6f, validation f1 = %.6f\n",
					elapsedTime, lambdas[bestLambdaIndex], maxF1);
			System.out.printf("Finished training the first %d lines of %s\n", lineCount+1, verbToTrainPath);
			line = verbToTrainReader.readLine();
			lineCount++;
		}

		verbToTrainReader.close();
	}
}
