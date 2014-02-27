package util;

import java.io.*;
import java.util.*;

public class MCMC {
	/**
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		String inputDir = args[0];
		if(!inputDir.endsWith("/")) {
			inputDir += "/";
		}
		// TODO Auto-generated method stub
		int T = 100;
		double gamma = 0.01;
		double alpha = 50.0 / T;

		int burnin = 0;
		int numIter = 20;
		int lag = 1;
		int numSampsPerLag = 1;
		int totNumSamps = 0;

		/*
		 * ###################################### # DATA INGESTION
		 * ######################################
		 */
		BufferedReader br = new BufferedReader(new FileReader(inputDir + "verbIdx.txt"));
		List<String> verbIdx = new ArrayList<String>();

		try {
			String line = br.readLine();

			while (line != null) {
				verbIdx.add(line);
				line = br.readLine();
			}
		} finally {
			br.close();
		}

		int V = verbIdx.size();

		BufferedReader brNoun = new BufferedReader(new FileReader(inputDir + "nounIdx.txt"));
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

		int C = 6594702;

		System.out.println("start training with " + C + " tuples in vnIdx...");

		/*
		 * ###################################### # MARKOV CHAIN INITIALIZATION
		 * ######################################
		 */
		// For each (v,n) pair with index i, z[i] will store an int from 1 to T,
		// telling us the current topic assignment of this pair
		int[] z = new int[C];

		// cumBeta and cumTheta will store the accumulated distro vectors, and be
		// used to average
		// (in the sense of Monte Carlo) once the Markov Chain stabilizes
		double[][] cumBeta = new double[T][N];
		double[][] cumTheta = new double[V][T];

		// #C^{WT}_{n,t} tells us the number of times a noun n has been assigned to
		// topic t
		int[][] CWT = new int[N][T];

		// C^{VT}_{v,t} tells us the number of times a topic t has been assigned to
		// the nouns that ever appear in a (v,n) pair
		int[][] CVT = new int[V][T];

		// tells us the number of nouns that have been assigned to each topic (i.e.
		// the column sum of CWT)
		int[] numNounsInTopic = new int[T];

		// tells us the number of nouns associated with each verb (i.e. the row sum
		// of CVT)
		int[] numNounsWithVerb = new int[V];

		// file connection for the vnIdx file
		BufferedReader brVn = new BufferedReader(new FileReader(inputDir + "vnIdxSmall.txt"));

		Random r = new Random();
		for (int i = 0; i < C; i++) {
			int t = r.nextInt(T); // randomly choose t
			z[i] = t;

			if (i % 10000 == 0)
				System.out.println("assigning tuple " + i + " to topic " + t);

			String nextLine = brVn.readLine();
			if (nextLine == null)
				System.err.println("read past the end of vnIdx");

			String[] vnPair = nextLine.split(",");
			int curV = Integer.parseInt(vnPair[0]);
			int curN = Integer.parseInt(vnPair[1]);

			CWT[curN][t] = CWT[curN][t] + 1;
			CVT[curV][t] = CVT[curV][t] + 1;

			numNounsInTopic[t] = numNounsInTopic[t] + 1;
			numNounsWithVerb[curV] = numNounsWithVerb[curV] + 1;
		}
		brVn.close();

		/*
		 * ###################################### # COLLAPSED GIBBS SAMPLING
		 * ######################################
		 */
		for (int iter = 0; iter < numIter; iter++) {
			System.out.println("At iteration " + iter);

			// file connection for the vnIdx file
			brVn = new BufferedReader(new FileReader(inputDir + "vnIdxSmall.txt"));

			for (int i = 0; i < C; i++) {

				int tau = z[i]; // the current topic assignment for this pair

				String nextLine = brVn.readLine();
				if (nextLine == null)
					System.err.println("read past the end of vnIdx");

				String[] vnPair = nextLine.split(",");
				int curV = Integer.parseInt(vnPair[0]);
				int curN = Integer.parseInt(vnPair[1]);

				if (CWT[curN][tau] == 0)
					System.err.println("warning: subtracting zero entry CWT[" + curN
					    + "," + tau + "]");
				CWT[curN][tau] = CWT[curN][tau] - 1;

				if (CVT[curV][tau] == 0)
					System.err.println("warning: subtracting zero entry CVT[" + curV
					    + "," + tau + "]");
				CVT[curV][tau] = CVT[curV][tau] - 1;

				numNounsInTopic[tau] = numNounsInTopic[tau] - 1;
				numNounsWithVerb[curV] = numNounsWithVerb[curV] - 1;

				double[] resampleDistro = new double[T];

				for (int t = 0; t < T; t++) {
					double pnt = (double) (CWT[curN][t] + gamma)
					    / (numNounsInTopic[t] + N * gamma);
					double ptv = (double) (CVT[curV][t] + alpha)
					    / (numNounsWithVerb[curV] + T * alpha);
					resampleDistro[t] = pnt * ptv;
				}

				tau = StatUtil.sampleMultinomial(resampleDistro);

				CWT[curN][tau] = CWT[curN][tau] + 1;
				CVT[curV][tau] = CVT[curV][tau] + 1;

				numNounsInTopic[tau] = numNounsInTopic[tau] + 1;
				numNounsWithVerb[curV] = numNounsWithVerb[curV] + 1;

				z[i] = tau;
			}

			if (iter > burnin && (iter % lag) < numSampsPerLag) {
				// only sample when iter = k*lag, k*lag+1, ..., k * lag +
				// (numSampsPerLag - 1)
				for (int t = 0; t < T; t++) {
					for (int n = 0; n < N; n++) {
						cumBeta[t][n] = cumBeta[t][n] + (CWT[n][t] + gamma)
						    / (numNounsInTopic[t] + N * gamma);
					}
				}

				for (int v = 0; v < V; v++) {
					for (int t = 0; t < T; t++) {
						cumTheta[v][t] = cumTheta[v][t] + (CVT[v][t] + alpha)
						    / (numNounsWithVerb[v] + T * alpha);
					}
				}

				totNumSamps = totNumSamps + 1;
			}
			brVn.close();
		}

		double[][] beta = new double[T][N];
		double[][] theta = new double[V][T];

		PrintWriter printWriter = new PrintWriter(inputDir + "beta.txt");

		for (int t = 0; t < T; t++) {
			for (int n = 0; n < N; n++) {
				beta[t][n] = cumBeta[t][n] / totNumSamps;

				printWriter.print(beta[t][n]);
				if (n < N - 1) {
					printWriter.print("\t");
				} else {
					printWriter.println();
				}
			}
		}

		printWriter.close();

		printWriter = new PrintWriter(inputDir + "theta.txt");
		for (int v = 0; v < V; v++) {
			for (int t = 0; t < T; t++) {
				theta[v][t] = cumTheta[v][t] / totNumSamps;

				printWriter.print(theta[v][t]);
				if (t < T - 1) {
					printWriter.print("\t");
				} else {
					printWriter.println();
				}
			}
		}

		printWriter.close();
	}
}
