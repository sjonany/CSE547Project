package learning.libsvm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import jnisvmlight.SVMLightModel;

/**
 * Investigate the weights of each model
 */
public class SvmlightModelViewer {
	private static final String MAPPING_FILENAME_TEMPLATE = "%s/mapping.txt";
	private static final String MODEL_FILENAME_TEMPLATE = "%s/model_%d.txt";
	
	// Top K most weighted features to show for each model
	private static final int TOP_K = 20;
	public static void main(String[] args) throws Exception {
		if(args.length != 1) {
			System.err.println("Usage: <models dir - must be a folder, here all the models + mapping reside>");
			return;
		}
		
		String modelDir = args[0];
		if(modelDir.endsWith("/")) {
			modelDir = modelDir.substring(0, modelDir.length()-1);
		}
		
		List<String> idToVerb = new ArrayList<String>();
		// make ids one-based
		idToVerb.add(null);
		BufferedReader mappingReader = new BufferedReader(new FileReader(String.format(MAPPING_FILENAME_TEMPLATE, modelDir)));
		String line = mappingReader.readLine();
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = toks[1];
			idToVerb.add(verb);
			line = mappingReader.readLine();
		}
		mappingReader.close();
		
		// -1 because I added a filler element
		int numVerb = idToVerb.size() - 1;
		for(int verbId = 1; verbId <= numVerb; verbId++) {
			String file = String.format(MODEL_FILENAME_TEMPLATE, modelDir, verbId);
			if(!new File(file).exists()) {
				continue;
			}
			SVMLightModel model = SVMLightModel.readSVMLightModelFromURL(new URL("file:" + file));
			final double[] weights = model.getLinearWeights();
			
			// Sort the features in descending weight magnitude
			List<Integer> indices = new ArrayList<Integer>();
			for(int i = 1; i <= numVerb; i++) {
				indices.add(i);
			}
			
			Collections.sort(indices, new Comparator<Integer>() {
				@Override
				public int compare(Integer arg0, Integer arg1) {
					return -Double.compare(weights[arg0], weights[arg1]);
				}
			});
			
			System.out.println(verbId + ". Model for verb = " + idToVerb.get(verbId));
			for(int k = 1; k <= TOP_K; k++) {
				int featIndex = indices.get(k);
				String featureName = "";
				if(featIndex >= idToVerb.size() ) {
					// zero based index of semantic feature
					featureName = "Semantic_" + (featIndex - 1 - idToVerb.size());
				} else {
					featureName = idToVerb.get(featIndex);
				}
				System.out.printf("\t%d. %s : %.6f\n", k, featureName, weights[featIndex]);
			}
		}
	}
}
