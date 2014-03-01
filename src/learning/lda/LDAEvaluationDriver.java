package learning.lda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import util.StatUtil;
import util.StatUtil.ClassificationPerformance;

/**
 * Driver for testing LDA model on dataset.
 */
public class LDAEvaluationDriver {
	public static void main(String[] args) throws Exception {
		String modelDir = args[0];
		String testPath = args[1];
		/** genCorpus will be used to compute P(D) and P(a1| D) */
		String genCorpusPath = args[2];

		LDAModel model = LDAModel.loadModel(modelDir);
		// Precompute the prior infos needed to convert LDA judgment to discriminative
		// count(noun ^ distractor) 
		Map<String, Integer> countNAndD = new HashMap<String, Integer>();
		long countDistractor =  0;
		long countData = 0;
		
		BufferedReader genReader = new BufferedReader(new FileReader(genCorpusPath));
		String line = genReader.readLine();
		int lineCount = 0;
		while(line != null) {
    	String[] toks = line.split("\t");
    	String obj = toks[1];
    	// TODO: How to compute P(a1|D) ? since frequency is undefined? I just treat all freqs as 1 then.
    	// int freq = Integer.parseInt(toks[2]);
    	boolean isPositive = Integer.parseInt(toks[3]) == 1;
    	int freq = 1;
    	if(!isPositive) {
    		countDistractor += freq;
    		StatUtil.addToTally(countNAndD, obj, freq);
    	}
    	countData += freq;
			line = genReader.readLine();
			lineCount++;
			if(lineCount % 100000 == 0) {
				System.out.printf("Processed %d lines from generalization corpus. \n", lineCount);
			}
		}
		Map<String, Double> prNGivenD = new HashMap<String, Double>();
		for(Entry<String, Integer> entry : countNAndD.entrySet()) {
			prNGivenD.put(entry.getKey(), 1.0 * entry.getValue() / countDistractor);
		}
		double prD = 1.0 * countDistractor / countData;
		genReader.close();
		System.out.println("Finished precomputing priors from generalization corpus.");
		
		BufferedReader testReader = new BufferedReader(new FileReader(testPath));
		lineCount = 0;
	  ClassificationPerformance result = new ClassificationPerformance();
		line = testReader.readLine();
    while(line != null) {
    	lineCount++;
    	if(lineCount % 100000 == 0)
    		System.out.println(lineCount);
    	String[] toks = line.split("\t");
    	String verb = toks[0];
    	
    	String obj = toks[1];
    	boolean isPositiveGold = Integer.parseInt(toks[3]) == 1;
    	boolean isPositivePrediction = predictIsDistractor(model, verb, obj, prNGivenD, prD);
    	
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
	
	private static boolean predictIsDistractor(LDAModel model, String verb, String obj, Map<String, Double> prNGivenD,
			double prD) {
		double pLda = 0.0;
		for(int topic = 0; topic < model.getTopicCount(); topic++) {
			pLda += model.getPrNounForTopic(obj, topic) * model.getPrTopicForVerb(topic, verb);
		}
		double prN_D = prNGivenD.containsKey(obj) ? prNGivenD.get(obj) : 0.0;
		double num = (1.0 - prD)  * pLda;
		double den = prD * prN_D + num;
		return num / den > 0.5;
	}
}
