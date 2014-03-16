package learning.lda;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.math3.special.Gamma;

public class StochasticVariationalInference {
	
	// Using the notations from Figure 6 of http://arxiv.org/pdf/1206.7051v3.pdf
	public static void main(String[] args) throws Exception {
		String baseDir = args[0];
		if(!baseDir.endsWith("/")) {
			baseDir += "/";
		}
		
		/*######################################
		# DATA INGESTION
		######################################*/
		BufferedReader br = new BufferedReader(new FileReader(baseDir + "verbIdx.txt"));
		List<String> verbIdx = new ArrayList<String>();
		
		try{
			String line = br.readLine();
			
			while (line != null) {
				verbIdx.add(line);
				line = br.readLine();
			}
		} finally {
			br.close();
		}
		
		int V = verbIdx.size();

		BufferedReader brNoun = new BufferedReader(new FileReader(baseDir + "nounIdx.txt"));
		List<String> nounIdx = new ArrayList<String>();
		
		try {
			String nounLine = brNoun.readLine();
			
			while (nounLine != null) {
				nounIdx.add(nounLine);
				nounLine = brNoun.readLine();
			}
		} finally {
			brNoun.close();
		}
		
		int N = nounIdx.size();				
		
		// file connection for the vnIdx file
		BufferedReader vnReader = new BufferedReader(new FileReader(baseDir + "vnIdxSmall.txt"));
		
		// Compress the verb id to be from 0->V
		Map<Integer, Integer> textIdToCompactId = new HashMap<Integer, Integer>();
		String line = vnReader.readLine();
		int numDistinctVerbs = 0;
		while(line != null) {
			String[] vnPair = line.split(",");
			int vid = Integer.parseInt(vnPair[0]);
			int nid = Integer.parseInt(vnPair[1]);
			
			if(!textIdToCompactId.containsKey(vid)) {
				int compactId = textIdToCompactId.size();
				textIdToCompactId.put(vid, compactId);
				numDistinctVerbs++;
			}
			
			line = vnReader.readLine();
		}
		
		System.out.println("There are a total of " + numDistinctVerbs + " distinct verbs in the dataset.");
		int[] compactIdToTextId = new int[numDistinctVerbs];
		for(Entry<Integer, Integer> entry : textIdToCompactId.entrySet()) {
			compactIdToTextId[entry.getValue()] = entry.getKey();
		}
		
		// dataset[i] = the dataset for verb with compact id i
		// dataset is a list of nouns n such that (v,n) appeared in corpus
		List<Integer>[] dataset = (List<Integer>[]) Array.newInstance(List.class, numDistinctVerbs);
		for(int i = 0; i < dataset.length; i++) {
			dataset[i] = new ArrayList<Integer>();
		}
		
		vnReader.close();
		vnReader = new BufferedReader(new FileReader(baseDir + "vnIdxSmall.txt"));
		line = vnReader.readLine();
		while(line != null) {
			String[] vnPair = line.split(",");
			int vid = Integer.parseInt(vnPair[0]);
			int nid = Integer.parseInt(vnPair[1]);
			
			int compactId = textIdToCompactId.get(vid);
			dataset[compactId].add(nid);
			line = vnReader.readLine();
		}
		System.out.println("Finished preprocessing training data...");

		////////////////////////////////////////////////////
		// Training
		

		final int NUM_TOPIC = 300;
		final double STEP_SIZE = 0.001;
		final double LAMBDA_MAGNITUDE = 1.0;
		final double ALPHA = 50.0 / NUM_TOPIC;
		final double ETA = 0.01;
		final int D = numDistinctVerbs;
		final int NUM_ROUNDS = 5;
		
		final double PHI_CONVERGENCE = 0.01;
		final double GAMMA_CONVERGENCE = 0.001;
		
		//lambda is param for distribution of objects/noun | topic
		double[][] lambda = new double[NUM_TOPIC][N];
	
		//Initialize lambda(t=0) randomly
		for(int k = 0; k < NUM_TOPIC; k++) {
			for(int n = 0; n < N; n++) {
				lambda[k][n] = Math.random() * LAMBDA_MAGNITUDE;
			}
		}
		
		// gamma_d_k is param for distribution of topic | document
		double[][] gamma = new double[numDistinctVerbs][NUM_TOPIC];
		
		// repeat until forever, streaming data points
		// for now, just round robin
		for(int iter = 0; iter < NUM_ROUNDS * numDistinctVerbs; iter++) {
			// Get a document id from the dataset
			int compactVerbId = iter % numDistinctVerbs;
			int d = compactVerbId;
			
			// init gamma
			for(int k = 0; k < NUM_TOPIC; k++) {
				gamma[d][k] = 1;
			}
			
			// precompute digamma of lambdas
			double[][] digamma_lambda = new double[NUM_TOPIC][N];
			double[] sum_digamma_lambda = new double[NUM_TOPIC];
			for(int k = 0; k < NUM_TOPIC; k++) {
				for(int n = 0; n < N; n++) {
					digamma_lambda[k][n] = Gamma.digamma(lambda[k][n]);
					sum_digamma_lambda[k] += digamma_lambda[k][n];
				}
			}
			
			// init phi_d - just used for storage
			double[][] phi = new double[dataset[d].size()][NUM_TOPIC];
			
			int numPhiGammaIter = 0;
			// compute phi and gamma until convergence
			while(true) {
				// precompute the digamma of gammas
				//digamma_gamma[k] = digamma(gamma_d_k)
				double[] digamma_gamma = new double[NUM_TOPIC];
				// sum_j=1->K {digamma(gamma_d_j)}
				double sum_digamma = 0.0;
				for(int k = 0; k < NUM_TOPIC; k++) {
					digamma_gamma[k] = Gamma.digamma(gamma[d][k]);
					sum_digamma += digamma_gamma[k];
				}
				
				double[][] tempPhi = new double[dataset[d].size()][NUM_TOPIC];
			
				// break the gamma_d = alpha + sum {phi_d_n}
				double[] tempGammaD = new double[NUM_TOPIC];
				Arrays.fill(tempGammaD, ALPHA);
				for(int i = 0; i < dataset[d].size(); i++) {
					int n = dataset[d].get(i);
					// compute and make sure sum {phi_dn} = 1
					for(int k = 0; k < NUM_TOPIC; k++) {
						double E_log_theta = digamma_gamma[k] - sum_digamma;
						double E_log_beta = digamma_lambda[k][n] - sum_digamma_lambda[k];
						tempPhi[i][k] = Math.exp(E_log_theta + E_log_beta);
					}
					
					// normalize phi's
					double normalizer = 0.0;
					for(int k = 0; k < NUM_TOPIC; k++) {
						normalizer += tempPhi[i][k];
					}
					
					for(int k = 0; k < NUM_TOPIC; k++) {
						tempPhi[i][k] /= normalizer;
					}
					
					// update gamma_d
					for(int k = 0; k < NUM_TOPIC; k++) {
						tempGammaD[k] += tempPhi[i][k];
					}
				}

				// check if phi_dn and gamma_d converge
				double eucDistGammaD = Math.sqrt(sqDiff(gamma[d], tempGammaD));
				double eucDistPhi = Math.sqrt(sqDiff(phi, tempPhi));
				if(numPhiGammaIter != 0) {
					if(eucDistGammaD < GAMMA_CONVERGENCE && eucDistPhi < PHI_CONVERGENCE) {
						break;
					}
				}
				
				gamma[d] = tempGammaD;
				phi = tempPhi;
				numPhiGammaIter++;
				if(numPhiGammaIter % 100 == 0) {
					System.out.printf("Num Phi Gamma Iter = %d, eucDistGammaD = %.6f, eucDistPhi = %.6f\n", 
							numPhiGammaIter, eucDistGammaD, eucDistPhi);
				}
			}// got phi and gamma
			
			// lambda_t = (1-p_t) * prev lambda + stuff ---- + stuff later
			for(int k = 0; k < NUM_TOPIC; k++) {
				for(int n = 0; n < N; n++) {
					lambda[k][n] *= 1.0 - STEP_SIZE;
				}
				
				// calculate lambda_prime
				double[] lambda_prime = new double[N];
				
				for(int i = 0; i < dataset[d].size(); i++) {
					lambda_prime[dataset[d].get(i)] += phi[i][k];
				}
				
				for(int n = 0; n < N; n++) {
					lambda_prime[n] *= D;
					lambda_prime[n] += ETA;
				}
				
				// merge lambda_prime
				for(int n = 0; n < N; n++) {
					lambda[k][n] += STEP_SIZE * lambda_prime[n];
				}
			} // done getting lambda(t)
		  
			// store params
			PrintWriter printWriter = new PrintWriter(baseDir + "betaAtIter" + iter + ".txt");
		
	    for (int t = 0; t < NUM_TOPIC; t++) {
	      for (int n = 0; n < N; n++) {
	        printWriter.print(lambda[t][n]);
	        if (n < N-1) {
	        	printWriter.print("\t");
	        } else {
	        	printWriter.println();
	        }
	      }
	    }
	    
	    printWriter.close();

	    printWriter = new PrintWriter(baseDir + "thetaAtIter" + iter + ".txt");
	    for (int v = 0; v < V; v++) {
	      for (int t = 0; t < NUM_TOPIC; t++) {
	      	//for verbs we don't care about, just put whatever.
	      	double theta_v_t = -1234.0;
	      	if(textIdToCompactId.containsKey(v)) {
	      		theta_v_t = gamma[textIdToCompactId.get(v)][t];
	      	}
	        printWriter.print(theta_v_t);
	        if (t < NUM_TOPIC-1) {
	        	printWriter.print("\t");
	        } else {
	        	printWriter.println();
	        }
	      }
	    }
	    
	    printWriter.close();
		} // for each data point = document    
	}//end main
	
	static double sqDiff(double[] a1, double[] a2) {
		if(a1.length != a2.length) {
			throw new IllegalArgumentException("The lengths of a1 and a2 don't match");
		}
		double sum = 0.0;
		for(int i = 0; i < a1.length; i++) {
			double diff = a1[i] - a2[i];
			sum += diff * diff;
		}
		
		return sum;
	}
	
	static double sqDiff(double[][] a1, double[][] a2) {
		if(a1.length != a2.length || a1[0].length != a2[0].length) {
			throw new IllegalArgumentException("The lengths of a1 and a2 don't match");
		}

		double sum = 0.0;
		for(int i = 0; i < a1.length; i++) {
			sum += sqDiff(a1[i], a2[i]);
		}
		
		return sum;
	}
	
}

