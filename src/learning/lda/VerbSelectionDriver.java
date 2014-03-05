package learning.lda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Driver for seeing the LDA model in a more interpretable way.
 */
public class VerbSelectionDriver {
	private static final int TOP_K = 10;
	
	private static final int[] TOPICS = {8,9,27,40,43,48};
	public static void main(String[] args) throws Exception {
		String modelDir = "/Users/sjonany/NELResources/SelectionalPref/gibbsSampling/topic_300_model";
		String vnIdxPath = "/Users/sjonany/NELResources/SelectionalPref/gibbsSampling/topic_300_model/vnIdx.txt";
		
		// load relevant verbs
		BufferedReader in = new BufferedReader(new FileReader(vnIdxPath));
		Set<Integer> relevantVerbIds = new HashSet<Integer>();
		String line = in.readLine();
		while(line != null) {
			relevantVerbIds.add(Integer.parseInt(line.split(",")[0])-1);
			line = in.readLine();
		}
		in.close();
		
		LDAModel model = LDAModel.loadModel(modelDir);
		
		int[] verbIdToBestTopicId = new int[model.getVerbCount()];
		for(int verbId : relevantVerbIds) {
			double maxProb = -1;
			int bestTopic = -1;
			for(int t = 0; t < model.getTopicCount(); t++) {
				double pr = model.getPrTopicForVerb(t, verbId);
				if(pr > maxProb) {
					maxProb = pr;
					bestTopic = t;
				}
			}
			verbIdToBestTopicId[verbId] = bestTopic;
		}
		
		for(int verbId : relevantVerbIds) {
			System.out.println("Verb = " + model.getVerb(verbId));
			System.out.println("Best topic = " + verbIdToBestTopicId[verbId] +"\n");
		}
		
		/*
		for(int topic : TOPICS) {
			System.out.println("Topic = " + topic);
			for(int verbId : relevantVerbIds) {
				if(verbIdToBestTopicId[verbId] == topic) {
					System.out.println("\t" + model.getVerb(verbId) + "\n");
				}
			}
		}*/
	}
}
