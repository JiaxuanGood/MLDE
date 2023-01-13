package myStack;

import java.io.File;
import java.io.FileOutputStream;

public class Logs {
	public static void prtResult(double[] result, boolean seq, String exp) {
		System.out.println("============" + exp + "============");
		String[] measures = {};
		String[] measures1 = {"acc", "precision", "recall", "F1-measure", "hamming loss", "hitrate", "micro_acc", "micro_pre", "micro_rec", "micro_f1", "micro_ham"};
		String[] measures2 = {"acc", "precision", "recall", "F1-measure", "hamming loss", "hitrate", "micro_acc", "micro_pre", "micro_rec", "micro_f1", "micro_ham", "one-error", "coverage", "rankingloss", "avgprecision"};
		if(seq)
			measures = measures2;
		else
			measures = measures1;
		
		for(int i=0; i<result.length; i++) {
			System.out.println(exp + "\t" + measures[i] + ":" + '\t' + result[i]);
		}
	}
	
	public static void prtResult(double[] result, double time, String dataName, String exp) throws Exception {
		System.out.println("============" + exp + "============");
		String[] measures2 = {"acc", "precision", "recall", "F1-measure", "hamming loss", "hitrate", "micro_acc", "micro_pre", "micro_rec", "micro_f1", "micro_ham", "one-error", "coverage", "rankingloss", "avgprecision"};
		for(int i=0; i<result.length; i++) {
			System.out.println(exp + "\t" + measures2[i] + ":" + '\t' + result[i]);
		}
		System.out.println(exp + "\ttime:\t" + time);
		setMatrix(result, time, dataName+'\t'+exp);
	}
	
	private static void setMatrix(double[] data, double time, String dataName) throws Exception {
		File file = new File("E:\\result.txt");
		FileOutputStream fileOutputStream = new FileOutputStream(file, true);
		fileOutputStream.write((dataName+"\t").getBytes());
		for(int i=0; i<data.length; i++) {
			fileOutputStream.write((data[i]+"\t").getBytes());
		}
		fileOutputStream.write((time+"\n").getBytes());
		fileOutputStream.close();
	}
	
	public static void prtAcc(double[] result, String exp) {
		System.out.println(exp + "\t" + "acc" + ":" + '\t' + result[0]);
	}
	
	
	/*
	 * Log out matrix
	 * these methods are mainly used in the process of programming
	 * */
	public static void logMatrix(double[][][] a, String exp) {
		System.out.println("============" + exp);
		for(int i=0; i<a.length; i++) {
			System.out.println("------------------------- " + "i=" + i + " -------------------------");
			for(int j=0; j<a[0].length; j++) {
				for(int k=0; k<a[0][0].length; k++) {
					System.out.print(a[i][j][k] + "\t");
				}
				System.out.println("");
			}
		}
	}
	
	public static void logMatrix(double[][] a, String exp) {
		System.out.println("============" + exp);
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[0].length; j++) {
				System.out.print(a[i][j] + "\t");
			}
			System.out.println("");
		}
	}
	
	public static void logMatrix(int[][] a, String exp) {
		System.out.println("============" + exp);
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[0].length; j++) {
				System.out.print(a[i][j] + "\t");
			}
			System.out.println("");
		}
	}
	
	public static void logMatrix(int[] a, String exp) {
		System.out.println("============" + exp);
		for(int i=0; i<a.length; i++) {
			System.out.println(a[i]);
		}
	}
	
	public static void logMatrix(double[] a, String exp) {
		System.out.println("============" + exp);
		for(int i=0; i<a.length; i++) {
			System.out.println(a[i]);
		}
	}
	
	public static void logMatrix_pan(double[] a) {
		for(int i=0; i<a.length; i++) {
			System.out.print("\t" + a[i]);
		}
		System.out.println();
	}
	public static void logMatrix_pan(double[] a, int b) {
		for(int i=0; i<b; i++) {
			System.out.print("\t" + a[i]);
		}
		System.out.println();
	}
	public static void logMatrix_pan(int[] a, int b) {
		for(int i=0; i<b; i++) {
			System.out.print("\t" + a[i]);
		}
		System.out.println();
	}
	public static void logMatrix_pan(int[] a) {
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i] + "\t");
		}
		System.out.println();
	}
	
	public static void logMatrix(boolean[] a, String exp) {
		System.out.println("============" + exp);
		for(int i=0; i<a.length; i++) {
			System.out.print(a[i] + " ");
		}
		System.out.println("");
	}
	
	public static void shape(double[][] a, String exp) {
		System.out.println(exp + "\t" + a.length + "x" + a[0].length);
	}
}
