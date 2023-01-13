package myStack;

import java.io.File;
import java.io.FileOutputStream;
import java.util.Arrays;

import mulan.data.MultiLabelInstances;
import weka.core.Instances;

public class Instruments {
	
	//split multiple data to several single data
	public static Instances[] multi2single(MultiLabelInstances mlData) {
		Instances data_all = mlData.getDataSet();
		int numLabel = mlData.getNumLabels();
		Instances[] datas = new Instances[numLabel];
		int[] r = mlData.getLabelIndices();
		Arrays.sort(r);
		for(int i=0; i<numLabel; i++) {
			datas[i] = new Instances(data_all);
			for(int j=numLabel-1; j>=0; j--) {
				if(i==j) {
					datas[i].setClassIndex(r[j]);
				}
				else {
					datas[i].deleteAttributeAt(r[j]);
				}
			}
			
		}
		return datas;
	}
	
	public static Instances multi2single(MultiLabelInstances mlData, int labelIndex) {
		Instances data_org = mlData.getDataSet();
		int numLabel = mlData.getNumLabels();
		Instances data_single = new Instances(data_org);
		int[] r = mlData.getLabelIndices();
		Arrays.sort(r);
		for(int j=numLabel-1; j>=0; j--) {
			if(labelIndex==j) {
				data_single.setClassIndex(r[j]);
			}
			else {
				data_single.deleteAttributeAt(r[j]);
			}
		}
		return data_single;
	}
	
	//thought from mulan.classifier.lazy.MLkNN
	public static int[][] getTarget(MultiLabelInstances dataset){
		Instances mInstances = dataset.getDataSet();
		int[][] t = new int[dataset.getNumInstances()][dataset.getNumLabels()];
		int[] labelIndices = dataset.getLabelIndices();
//		Logs.logMatrix(labelIndices, "labelIndices:");
		for (int i = 0; i < dataset.getNumInstances(); i++) {
			for (int j = 0; j < dataset.getNumLabels(); j++) {
//				double value = Double.parseDouble(mInstances.attribute(labelIndices[j]).value((int) mInstances.instance(i).value(labelIndices[j])));
//				t[i][j] = (int)value;
				t[i][j] = (int) mInstances.instance(i).value(labelIndices[j]);
			}
		}
//		Logs.logMatrix(t, "target matrix:");
		return t;
	}
	public static boolean[][] getTruth(MultiLabelInstances dataset){
		int[][] t = getTarget(dataset);
		boolean[][] t_bi = new boolean[t.length][t[0].length];
		for (int i = 0; i < t.length; i++) {
			for (int j = 0; j < t[0].length; j++) {
				t_bi[i][j]=t[i][j]==1;
			}
		}
		return t_bi;
	}
	
	public static int[] getTarget(Instances dataset){
		Instances mInstances = dataset;
		int[] t = new int[dataset.numInstances()];
		int labelIndices = dataset.classIndex();
		for (int i = 0; i < dataset.numInstances(); i++) {
//			double value = Double.parseDouble(mInstances.attribute(labelIndices).value((int) mInstances.instance(i).value(labelIndices)));
//			t[i] = (int)value;
			t[i] = (int) mInstances.instance(i).value(labelIndices);
		}
		return t;
	}
	
	public static double[][] getContent(Instances dataset){
		Instances mInstances = new Instances(dataset);
		double[][] atts = new double[dataset.numInstances()][dataset.numAttributes()];
		for (int i = 0; i < dataset.numInstances(); i++) {
			for (int j = 0; j < dataset.numAttributes(); j++) {
				atts[i][j] = (double) mInstances.instance(i).value(j);
			}
		}
		return atts;
	}
	
	public static int getNumFeat(MultiLabelInstances mli) {
		int numlabel = mli.getNumLabels();
		int numall = mli.getDataSet().numAttributes();
		return numall-numlabel;
	}
	
	public static int[] getTopM3(double[] seq, int m) {
		int[] index = sort2(seq);
		int[] r = new int[m];
		for(int i=0; i<m; i++) {
			r[i] = index[i];
		}
		return r;
	}
	
	public static int[] sort2(double[] seq) {
		int[] b = new int[seq.length];
		double[] a = new double[seq.length];
		for(int i=0; i<seq.length; i++) {
			b[i] = i;
			a[i] = seq[i];
		}
		double tmp;
		int tmp2;
		for(int i=0; i<seq.length; i++) {
			for(int j=i+1; j<seq.length; j++) {
				if(a[i]<a[j]) {
					tmp = a[i];
					a[i] = a[j];
					a[j] = tmp;
					
					tmp2 = b[i];
					b[i] = b[j];
					b[j] = tmp2;
				}
			}
		}
		return b;
	}
	
	public static int[] getsort(double[] seq) {
		int numL = seq.length;
		double max = 0;
		for(int i=0; i<numL; i++) {
			if(seq[i]>max) {
				max = seq[i];
				if(max>0.99) {
					break;
				}
			}
		}
		int[] b = new int[numL+1];	//[numL+1]: the last for number of perfects
		int cnt = 0;
		for(int i=0; i<numL; i++) {
			if(1-seq[i]<0.0001) {
				b[cnt] = i;
				cnt++;
			}
		}
		if(cnt==0) {
			b[numL] = 1;
		}
		else {
			b[numL] = cnt;
		}
		return b;	//b is the index
	}
	
	public static void setMatrix(double[][] result) throws Exception {
		setMatrix(result, "result2");
	}
	public static void setMatrix(double[][] result, String filename) throws Exception {
		for(int i=0; i<result.length; i++) {
			setMatrix(result[i], filename);
		}
	}
	public static void setMatrix(double[] result) throws Exception {
		setMatrix(result, "result2");
	}
	public static void setMatrix(double[] result, String filename) throws Exception {
		File directory = new File("");
		String dirpath = directory.getAbsolutePath();
		File file = new File(dirpath + "\\"+filename+".txt");
		FileOutputStream fileOutputStream = new FileOutputStream(file, true);
		for(int i=0; i<result.length; i++) {
			fileOutputStream.write((result[i]+"\t").getBytes());
		}
		fileOutputStream.write("\n".getBytes());
		fileOutputStream.close();
	}
	
	public static double freeMem() {
		Runtime currRuntime = Runtime.getRuntime ();
		double nFreeMemory = currRuntime.freeMemory() / 1024 / 1024;
		return nFreeMemory;
	}
	
	public static String showMemoryInfo() {
		Runtime currRuntime = Runtime.getRuntime ();
		int nFreeMemory = ( int ) (currRuntime.freeMemory() / 1024 / 1024);
		int nTotalMemory = ( int ) (currRuntime.totalMemory() / 1024 / 1024);
		return nFreeMemory + "M/" + nTotalMemory +"M(free/total)" ;
	}
	
	public static void main(String[] str) {
		double seq[] = {0.2708333333333333,0.7946428571428572,0.3441558441558441,0.7083333333333333,0.5705128205128205,0.45,0.4857142857142857,0.5625,
				0.5634920634920635,0.525,0.5857142857142856,0.3269230769230769,0.538961038961039,0.3484848484848485,0.525,0.39610389610389607,
				0.34285714285714286,0.9375,0.9333,0.4875,0.4675324675324675,0.6272727272727273,0.41666666666666663,0.4222222222222222,
				0.5317460317460317,0.16666666666666666,0.5833333333333333,0.5324675324675324,0.7222222222222222,0.6222222222222222,0.4920634920634921,
				0.5056818181818181,0.5555555555555556,0.4097222222222222,0.4097222222222222,0.6388888888888888,0.4318181818181818,0.325,0.7708333333333333,
				0.4494949494949495,0.625,0.6349206349206349,0.4857142857142857,0.35,0.6363636363636364,0.5142857142857142,0.5,0.32386363636363635,
				0.44155844155844154,0.5606060606060606};
//		double seq[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
//				1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
//		double seq[] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
//				1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
//		double seq[] = {1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8};
		
		Logs.logMatrix(seq, "orig");
//		int[] r = sort(seq);
//		Arrays.sort(seq);
//		Logs.logMatrix(seq, "sort");
//		Logs.logMatrix(r, "index");
		
	}
}
