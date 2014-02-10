package data.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import util.StatUtil;

/**
 * Create train, test, validation set given output from PositiveNegativeDatasetCreator
 * Make sure to -SHUFFLE- the dataset before hand!
 * Proportions are : 95% train, 2.5% validation, 2.5% test
 */
public class TrainTestValidationCreator {
	private static double TRAIN_PERCENT = 0.95;
	private static double TEST_PERCENT = 0.025;
	private static double VALIDATION_PERCENT = 1.0 - TRAIN_PERCENT - TEST_PERCENT;
	
	public static void main(String[] args) throws Exception {
		if(args.length != 3) {
			System.err.println("Usage: <input file> <output file prefix (will add .train .test. validation later)> " +
					"<min cluster size: int - verbs with less than cluster size will be discarded>");
			return;
		}
		
		String inputFile = args[0];
		String outputFile = args[1];
		int MIN_CLUSTER_SIZE = Integer.parseInt(args[2]);
		
		// Compute number of training instances for each verb
		BufferedReader in = new BufferedReader(new FileReader(inputFile));
		Map<String, Integer> verbToClusterSize = new HashMap<String, Integer>();
		String line = in.readLine();
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = toks[0];
			StatUtil.addToTally(verbToClusterSize, verb, 1);
			line = in.readLine();
		}
		
		System.out.printf("There are %d distinct verbs in total\n", verbToClusterSize.size());
		in.close();
		
		// Go through all the rows again, and separate them into 3 partitions
		PrintWriter trainWriter = new PrintWriter(outputFile + ".train");
		PrintWriter testWriter = new PrintWriter(outputFile + ".test");
		PrintWriter validationWriter = new PrintWriter(outputFile + ".validation");
		in = new BufferedReader(new FileReader(inputFile)); 
		
		Map<String, Integer> verbToNumWritten = new HashMap<String, Integer>();
		int numTrainRow = 0;
		int numTestRow = 0;
		int numValidationRow = 0;
		
		line = in.readLine();
		while(line != null) {
			String[] toks = line.split("\t");
			String verb = toks[0];
			
			// if this verb has too few training examples, skip it
			int verbClusterSize = verbToClusterSize.get(verb); 
			if(verbClusterSize < MIN_CLUSTER_SIZE) {
				line = in.readLine();
				continue;
			}
			
			Integer numWritten = verbToNumWritten.get(verb);
			if(numWritten == null) numWritten = 0;
			if(numWritten <= TRAIN_PERCENT * verbClusterSize) {
				trainWriter.println(line);
				numTrainRow++;
			} else if(numWritten <= (TRAIN_PERCENT + TEST_PERCENT) * verbClusterSize) {
				testWriter.println(line);
				numTestRow++;
			} else {
				validationWriter.println(line);
				numValidationRow++;
			}
			
			StatUtil.addToTally(verbToNumWritten, verb, 1);
			line = in.readLine();
		}

		System.out.printf("There are %d verb clusters used in the end\n", verbToNumWritten.size());
		System.out.printf("Num train: %d, Num test: %d, Num validation: %d\n", numTrainRow, numTestRow, numValidationRow);
		in.close();
		trainWriter.close();
		testWriter.close();
		validationWriter.close();		
	}
}
