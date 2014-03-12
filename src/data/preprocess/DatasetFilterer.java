package data.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashSet;
import java.util.Set;

/**
 * Filters a dataset to just a subset of verbs -> I should have done this long time ago.
 * @author sjonany
 *
 */
public class DatasetFilterer {
	public static void main(String[] args) throws Exception {
		String datasetFile = args[0];
		String pathToVerbSubset = args[1];	
		String outputFile = args[2];
		
		Set<String> verbSubset = new HashSet<String>();
		BufferedReader verbToTestReader = new BufferedReader(new FileReader(pathToVerbSubset));
		String line = verbToTestReader.readLine();
		
		while(line != null) {
			// Perform test for each verb
	  	String verbString = line.split("\t")[0];
	  	verbSubset.add(verbString);
	  	line = verbToTestReader.readLine();
	  	
		}
		verbToTestReader.close();
	
		BufferedReader datasetReader = new BufferedReader(new FileReader(datasetFile));
		PrintWriter datasetWriter = new PrintWriter(outputFile);
		
		line = datasetReader.readLine();
		while(line != null) {
			String verb = line.split("\t")[0];
			if(verbSubset.contains(verb)) {
				datasetWriter.println(line);
			}
			line = datasetReader.readLine();
		}
		
		datasetReader.close();
		datasetWriter.close();
	}	
}
