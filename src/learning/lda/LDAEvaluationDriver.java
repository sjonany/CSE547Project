package learning.lda;

import java.io.BufferedReader;
import java.io.FileReader;

import util.StatUtil.ClassificationPerformance;

/**
 * Driver for testing LDA model on dataset.
 */
public class LDAEvaluationDriver {
	public static void main(String[] args) throws Exception {
		String modelDir = args[0];
		String testPath = args[1];
		
		LDAModel model = LDAModel.loadModel(modelDir);
		BufferedReader testReader = new BufferedReader(new FileReader(testPath));
		int lineCount = 0;
	  ClassificationPerformance result = new ClassificationPerformance();
		String line = testReader.readLine();
    while(line != null) {
    	lineCount++;
    	if(lineCount % 100000 == 0)
    		System.out.println(lineCount);
    	String[] toks = line.split("\t");
    	String verb = toks[0];
    	
    	String obj = toks[1];
    	boolean isPositiveGold = Integer.parseInt(toks[3]) == 1;
    	boolean isPositivePrediction = predictIsDistractor(model, verb, obj);
    	
    	if(isPositiveGold) {
    		if(isPositivePrediction) {
    			result.tp++;
    		} else {
    			result.fn++;
    		}
    	} else {
    		if(isPositivePrediction) {
    			result.fp++;
    		} else {
    			result.tn++;
    		}
    	}
    	
    	line = testReader.readLine();
    }
		System.out.printf("Processed all! , acc = %d/%d = %.2f, precision = %.2f, recall = %.2f, f1 = %.2f\n",
		    result.tp + result.tn, lineCount, result.getAccuracy(), result.getPrecision(),
		    result.getRecall(), result.getFscore());
    
    testReader.close();
	}
	
	private static boolean predictIsDistractor(LDAModel model, String verb, String obj) {
		// TODO: figure out P(distractor), and P(noun | distractor)
		return true;
	}
}
