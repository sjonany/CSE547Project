package learning.libsvm;

import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.SVMLightModel;
import jnisvmlight.TrainingParameters;
import util.StatUtil.ClassificationPerformance;
import util.SvmlightUtil;

public class SingleModelExperiment {
	public static void main(String[] args) throws Exception { 
		if(args.length != 3 ) {
			System.err.println("<path to train> <path to test> <verb to train: string>");
			return;
		}
		
		String trainFile = args[0];
		String testFile = args[1];
		String targetVerb = args[2];

	  VerbObjectStatComputer stats = new VerbObjectStatComputer();
	  System.out.println("Precomputing stats from train file...");
	  stats.load(trainFile);
	  System.out.println("Finished precomputation.");
	  
	  FeatureExtractor featureExtractor = new VerbCooccurrenceFeatureExtractor(stats);
	  
	  System.out.println("Loading train set");
    LabeledFeatureVector[] trainSet = SvmlightUtil.filterDatasetToVerb(trainFile, targetVerb, featureExtractor);
    System.out.println("Loading test set");
    LabeledFeatureVector[] testSet = SvmlightUtil.filterDatasetToVerb(testFile, targetVerb, featureExtractor);    
    
	  double[] lambdas = {0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000.0 ,1000000000.0};
	  double[] costRatios = {1.0,1.5,2.0,5.0,10.0,50, 100};
	  System.out.println("Lambda\ttrainPrecision\ttrainRecall\ttrainF1\ttrainAccuracy\ttestPrecision\ttestRecall\ttestF1\ttestAccuracy");
		for(double lambda : lambdas) {
			for(double costRatio : costRatios) {
		    TrainingParameters trainParam = new TrainingParameters();
		    trainParam.getLearningParameters().svm_costratio = costRatio;
		    trainParam.getLearningParameters().svm_c = lambda;
		    SVMLightModel model = new SVMLightInterface().trainModel(
		    				trainSet, trainParam);
		    ClassificationPerformance trainResult = SvmlightUtil.testModel(model, trainSet);
		    ClassificationPerformance testResult = SvmlightUtil.testModel(model, testSet);
		    if(trainResult.tp == 0 || testResult.tp == 0) {
		    	continue;
		    }
		    System.out.printf("%.2f\t%.2f -- %.2f\t%.2f\t%.2f\t%.2f -- %.2f\t%.2f\t%.2f\t%.2f\n", lambda, costRatio, 
		    		trainResult.getPrecision(), trainResult.getRecall(), trainResult.getFscore(), trainResult.getAccuracy(),
		    		testResult.getPrecision(), testResult.getRecall(), testResult.getFscore(), testResult.getAccuracy());
			}
		}
	}
}
