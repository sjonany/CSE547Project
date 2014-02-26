package learning.lda;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

/**
 * Driver for seeing the LDA model in a more interpretable way.
 */
public class FeatureInterpretabilityDriver {
	private static final int TOP_K = 5;
	
	public static void main(String[] args) throws Exception {
		String modelDir = "/Users/sjonany/NELResources/SelectionalPref/dataset";
		LDAModel model = LDAModel.loadModel(modelDir);
		
		for(int topic = 0; topic < model.getTopicCount(); topic++) {
			List<Pair<Integer, Double>> nounPrPairs = new ArrayList<Pair<Integer, Double>>();
			for(int nounId = 0; nounId < model.getNounCount(); nounId++) {
				double pr = model.getPrNounForTopic(nounId, topic);
				nounPrPairs.add(Pair.of(nounId, pr));
			}
			Collections.sort(nounPrPairs, new Comparator<Pair<Integer, Double>>() {
				@Override
        public int compare(Pair<Integer, Double> arg0,
            Pair<Integer, Double> arg1) {
					return -Double.compare(arg0.getRight(), arg1.getRight());
        }
			});
			
			System.out.println("Topic " + topic + ": ");
			for(int i = 0; i < TOP_K; i++) {
				Pair<Integer, Double> pair = nounPrPairs.get(i);
				System.out.println("\t" + model.getNoun(pair.getLeft()) + ": " + pair.getRight());
			}
		}
	}
}
