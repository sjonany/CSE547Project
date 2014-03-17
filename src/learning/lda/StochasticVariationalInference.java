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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.distribution.GammaDistribution;

import util.StatUtil;

public class StochasticVariationalInference {
	
	// how many iterations til i start keeping a log of the params
	private static final int MIN_ITER_TO_TRACK = 0;

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
			int vid = Integer.parseInt(vnPair[0])-1;
			int nid = Integer.parseInt(vnPair[1])-1;
			
			if(!textIdToCompactId.containsKey(vid)) {
				int compactId = textIdToCompactId.size();
				textIdToCompactId.put(vid, compactId);
				numDistinctVerbs++;
			}
			
			line = vnReader.readLine();
		}
		
		System.out.println("There are a total of " + numDistinctVerbs + " distinct verbs in the dataset. = " + textIdToCompactId.keySet());
		int[] compactIdToTextId = new int[numDistinctVerbs];
		for(Entry<Integer, Integer> entry : textIdToCompactId.entrySet()) {
			compactIdToTextId[entry.getValue()] = entry.getKey();
		}
		// count of (v,n) in the corpus
		Map<Pair<Integer, Integer>, Integer> vnCount = new HashMap<Pair<Integer, Integer>, Integer>();
		
		vnReader.close();
		vnReader = new BufferedReader(new FileReader(baseDir + "vnIdxSmall.txt"));
		line = vnReader.readLine();
		while(line != null) {
			String[] vnPair = line.split(",");
			// The indices are one-based. I want zero-based
			int vid = Integer.parseInt(vnPair[0])-1;
			int nid = Integer.parseInt(vnPair[1])-1;
			
			int compactId = textIdToCompactId.get(vid);
			Pair p = Pair.of(compactId, nid);
			StatUtil.addToTally(vnCount, p, 1);
			line = vnReader.readLine();
		}
		
		// compressedDataset[i] contains samples for document (verb) i
		// each sample is a <nounId, frequency> pair
		List<Pair<Integer, Integer>>[] compressedDataset = 
				(List<Pair<Integer, Integer>>[]) Array.newInstance(List.class, numDistinctVerbs);;
		
		for(int i = 0; i < compressedDataset.length; i++) {
			compressedDataset[i] = new ArrayList();
		}
		
		for(Pair<Integer, Integer> vn : vnCount.keySet()) {
			int v = vn.getKey();
			int n = vn.getValue();
			int freq = vnCount.get(vn);
			compressedDataset[v].add(Pair.of(n, freq));
		}
		System.out.println("Finished preprocessing training data...");

		////////////////////////////////////////////////////
		// Training
		
		final int NUM_TOPIC = 300;
		// page 35
		final double ALPHA = 1.0 / NUM_TOPIC;
		final double ETA = 1.0 / NUM_TOPIC;
		
		final int D = numDistinctVerbs;
		final int NUM_ROUNDS = 1000;
		
		final double GAMMA_CONVERGENCE = 0.01 * NUM_TOPIC;
		final int MAX_CONVERGENCE_ITER = 100;
		
		//lambda is param for distribution of objects/noun | topic
		double[][] lambda = new double[NUM_TOPIC][N];

		GammaDistribution lambdaSampler = new GammaDistribution(100.0, 0.01);
		
		//Initialize lambda(t=0) randomly
		for(int k = 0; k < NUM_TOPIC; k++) {
			for(int n = 0; n < N; n++) {
				lambda[k][n] = lambdaSampler.sample();
			}
		}
		
		//maxNounPerVerb = 3,267,225
		
		// gamma_d_k is param for distribution of topic | document
		double[][] gamma = new double[numDistinctVerbs][NUM_TOPIC];
		
		// initialize all the matrices here to save memory
		double[][] digamma_lambda = new double[NUM_TOPIC][N];
		double[] sum_digamma_lambda = new double[NUM_TOPIC];
		double[] digamma_gamma = new double[NUM_TOPIC];
		double[] tempGammaD = new double[NUM_TOPIC];
		double[] pows = new double[NUM_TOPIC];
		
		// repeat until forever, streaming data points
		// for now, just round robin
		for(int iter = 0; iter < NUM_ROUNDS * numDistinctVerbs; iter++) {
			// page 1320 of http://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf
			double stepSize = Math.pow((iter + 1), -0.7);
			
			System.out.println("Starting sample " + iter);
			// Get a document id from the dataset
			int compactVerbId = iter % numDistinctVerbs;
			int d = compactVerbId;
			
			// init gamma
			// the paper says set to 1, but the code i found online sampled from gamma dist
			for(int k = 0; k < NUM_TOPIC; k++) {
				gamma[d][k] = lambdaSampler.sample();
			}
			
			// precompute digamma of lambdas
			clear(digamma_lambda);
			Arrays.fill(sum_digamma_lambda, 0);
			for(int k = 0; k < NUM_TOPIC; k++) {
				for(int n = 0; n < N; n++) {
					digamma_lambda[k][n] = digamma0(lambda[k][n]);
					sum_digamma_lambda[k] += lambda[k][n];
				}
				sum_digamma_lambda[k] = digamma0(sum_digamma_lambda[k]);
			}
		
			int numPhiGammaIter = 0;

			// calculate lambda_prime -- need to do it on the fly if we don't want to save phi's
			double[][] lambda_prime = new double[NUM_TOPIC][N];
			
			// compute phi and gamma until convergence
			while(true) {
				clear(lambda_prime);
				
				double curTime = System.currentTimeMillis();
				// precompute the digamma of gammas
				//digamma_gamma[k] = digamma(gamma_d_k)
				Arrays.fill(digamma_gamma, 0);
				// digamma(sum_j=1->K {gamma_d_j})
				double sum_digamma = 0.0;
				for(int k = 0; k < NUM_TOPIC; k++) {
					digamma_gamma[k] = digamma0(gamma[d][k]);
					sum_digamma += gamma[d][k];
				}
				sum_digamma = digamma0(sum_digamma);
			
				// break the gamma_d = alpha + sum {phi_d_n}
				Arrays.fill(tempGammaD, ALPHA);
				for(int i = 0; i < compressedDataset[d].size(); i++) {
					Pair<Integer, Integer> nFreq = compressedDataset[d].get(i);
					int noun = nFreq.getKey();
					int freq = nFreq.getValue();
					
					// the E_log_theta + E_log_beta's, we can't just compute e ^ that for the phi's
					Arrays.fill(pows, 0);
					// compute and make sure sum {phi_dn} = 1
					for(int k = 0; k < NUM_TOPIC; k++) {
						double E_log_theta = digamma_gamma[k] - sum_digamma;
						double E_log_beta = digamma_lambda[k][noun] - sum_digamma_lambda[k];
						pows[k] = Math.exp(E_log_theta + E_log_beta);
						// System.out.println("Stub");
					}
					
					double normalizer = 0.0;
					for(double p : pows) {
						normalizer += p;
					}
					
					// update gamma_d
					for(int k = 0; k < NUM_TOPIC; k++) {
						// Can collapse identical words in the document together, since the math form is exactly the same
						double phi_i_k = pows[k] / normalizer;
						tempGammaD[k] += freq * phi_i_k;
						
						// Assume that this is the converging iteration
						// if not converging iteration, lambda_prime gets reset to 0 anyways
						lambda_prime[k][noun] +=  freq * phi_i_k;
					}
				}

				//System.out.println("GammaD = " + Arrays.toString(tempGammaD));
				//System.out.println("Phi = " + Arrays.deepToString(tempPhi));
				
				// check if phi_dn and gamma_d converge
				double eucDistGammaD = Math.sqrt(sqDiff(gamma[d], tempGammaD));
				//double eucDistPhi = Math.sqrt(sqDiff(phi, tempPhi));

				//if(numPhiGammaIter % 100 == 0) {
					System.out.printf("Num Phi Gamma Iter = %d, eucDistGammaD = %.6f, timeElapsed(s) = %.6f\n", 
							numPhiGammaIter, eucDistGammaD, (System.currentTimeMillis() - curTime) / 1000.0);
				//}
					
				if(numPhiGammaIter != 0) {
					if(eucDistGammaD < GAMMA_CONVERGENCE || numPhiGammaIter > MAX_CONVERGENCE_ITER) {
						break;
					}
				}
				
				//System.out.println("tempGammaD = " + compress(tempGammaD));
				
				for(int i = 0; i < tempGammaD.length; i++) {
					gamma[d][i] = tempGammaD[i];
				}
				
				numPhiGammaIter++;
			}// got phi and gamma
			
			// lambda_t = (1-p_t) * prev lambda + stuff ---- + stuff later
			for(int k = 0; k < NUM_TOPIC; k++) {
				for(int n = 0; n < N; n++) {
					lambda[k][n] *= 1.0 - stepSize;
				}
				
				// lambda_prime will already have sum {phi w} term from the very last iteration - when converged
				
				for(int n = 0; n < N; n++) {
					lambda_prime[k][n] *= D;
					lambda_prime[k][n] += ETA;
				}
				
				// merge lambda_prime
				for(int n = 0; n < N; n++) {
					lambda[k][n] += stepSize * lambda_prime[k][n];
				}
			} // done getting lambda(t)		
			
			// store params

			// We have the free variables, but want to recover the parameters
			// We just get the mode -- beta ~ Dir(alpha), just get the beta that maxes this prob
			
			// logging
			if(iter > MIN_ITER_TO_TRACK && (iter % 10 == 0 || iter % 10 == 1)) {	
				PrintWriter printWriter = new PrintWriter(baseDir + "betaAtIter" + iter + ".txt");
			
		    for (int t = 0; t < NUM_TOPIC; t++) {
		    	double[] beta_t = getDirichletMode(lambda[t]);
		      for (int n = 0; n < N; n++) {
		        printWriter.print(beta_t[n]);
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
		    	double[] theta_v = new double[NUM_TOPIC];
	      	//for verbs we don't care about, just put whatever.
		    	Arrays.fill(theta_v, -1234.0);
	      	if(textIdToCompactId.containsKey(v)) {
	      		theta_v = getDirichletMode(gamma[textIdToCompactId.get(v)]);
	      	}
	      	
		      for (int t = 0; t < NUM_TOPIC; t++) {
		        printWriter.print(theta_v[t]);
		        if (t < NUM_TOPIC-1) {
		        	printWriter.print("\t");
		        } else {
		        	printWriter.println();
		        }
		      }
		    }
		    printWriter.close();

				printWriter = new PrintWriter(baseDir + "lambdaAtIter" + iter + ".txt");
			
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
		    
				printWriter = new PrintWriter(baseDir + "gammaAtIter" + iter + ".txt");
			
				for(int v = 0; v < numDistinctVerbs; v++) {
					printWriter.print(compactIdToTextId[v] + ": ");
			    for (int t = 0; t < NUM_TOPIC; t++) {
			       printWriter.print(gamma[v][t]);
			       if (t < NUM_TOPIC-1) {
			       	printWriter.print("\t");
			       } else {
			       	printWriter.println();
			       }
			    }
				}
		    	    
		    printWriter.close();
			} // end logging
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
	
	// Non-recursive digamma from http://www.cs.princeton.edu/~blei/topicmodeling.html
	// lda-c-dist/utils.c
	static double digamma0(double x)
	{
	    double p;
	    x=x+6;
	    p=1/(x*x);
	    p=(((0.004166666666667*p-0.003968253986254)*p+ 0.008333333333333)*p-0.083333333333333)*p;
	    p=p+Math.log(x)-0.5/x-1/(x-1)-1/(x-2)-1/(x-3)-1/(x-4)-1/(x-5)-1/(x-6);
	    return p;
	}
	
	// approx log {sum e^ki}
	public static double logSumExp(double[] pows){
		double max = maxArr(pows).getRight();
		double sum = 0.0;
		for(double p : pows) {
			sum += Math.exp(p - max);
		}
		return max + Math.log(sum);
	}
	
	public static Pair<Integer, Double> minArr(double[] arr) {
		double min = Double.POSITIVE_INFINITY;
		int minIndex = -1;
		for(int i = 0; i < arr.length; i++) {
			if(min > arr[i]) {
				min = arr[i];
				minIndex = i;
			}
		}
		return Pair.of(minIndex, min);
	}
	

	public static Pair<Integer, Double> maxArr(double[] arr) {
		double max = Double.NEGATIVE_INFINITY;
		int maxIndex = -1;
		for(int i = 0; i < arr.length; i++) {
			if(max < arr[i]) {
				max = arr[i];
				maxIndex = i;
			}
		}
		return Pair.of(maxIndex, max);
	}
	
	// see "mode" from http://en.wikipedia.org/wiki/Dirichlet_distribution
	public static double[] getDirichletMode(double[] dirParams) {
		double sum = 0.0;
		int K = dirParams.length;
		for(double x : dirParams) {
			sum += x;
		}
		
		double[] mode = new double[K];
		for(int i = 0; i < K; i++) {
			mode[i] = (dirParams[i] - 1.0) / (sum - K); 
		}
		
		return mode;
	}
	
	public static void clear(double[][] mat) {
		for(double[] arr : mat) {
			Arrays.fill(arr, 0);
		}
	}
	
	public static Map<Double, Integer> compress(double[] arr) {
		Map<Double, Integer> res = new HashMap<Double, Integer>();
		for(double x : arr) {
			StatUtil.addToTally(res, x, 1);
		}
		return res;
	}
	
}

