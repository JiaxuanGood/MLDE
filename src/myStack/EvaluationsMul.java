package myStack;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;

/*
 * author JiaxuanLi
 * */

public class EvaluationsMul {
	private double threshold = 0.5;
	private String[] Measures = {"accuracy", "precision", "recall", "F1_measure", "hitrate", "subsetACC", "hamming loss", "one_error", "coverage", "rankingloss", "avgprecision"};
	
	//Save evaluation results to local file
	private void setMatrix(double[] result, double[] times, String datalg) throws Exception {
		File directory = new File("");
		String dirpath = directory.getAbsolutePath();
		File file = new File(dirpath + "\\result.txt");
		FileOutputStream fileOutputStream = new FileOutputStream(file, true);
		fileOutputStream.write((datalg+"\t").getBytes());
		for(int i=0; i<result.length; i++) {
			fileOutputStream.write((result[i]+"\t").getBytes());
		}
		fileOutputStream.write((times[0]+"").getBytes());
		for(int i=1; i<times.length; i++) {
			if(times[i]>0) {
				fileOutputStream.write(("\t"+times[i]).getBytes());
			}
		}
		fileOutputStream.write("\n".getBytes());
		fileOutputStream.close();
	}
	
	//Show all evaluation results, and then save them to local file
	public void prtConclusion(double[] result, double time, String dataSet, String alg) throws Exception {
		System.out.println(dataSet + "\t" + alg);
		for(int i=0; i<result.length; i++) {
			System.out.println(Measures[i] + ":" + '\t' + result[i]);
		}
		System.out.println("time:\t" + time);
		double[] times = {time, 0};
		setMatrix(result, times, dataSet+'\t'+alg);
	}
	
	public void prtConclusion(double[] result, double[] times, String dataSet, String alg) throws Exception {
		System.out.println(dataSet + "\t" + alg);
		for(int i=0; i<result.length; i++) {
			System.out.println(Measures[i] + ":" + '\t' + result[i]);
		}
		System.out.print("time:");
		for(int i=0; i<times.length; i++) {
			System.out.print("\t" + times[i]);
		}
		System.out.println();
		setMatrix(result, times, dataSet+'\t'+alg);
	}
	
	public void prtConclusion(double[] result, String dataSet, String alg) throws Exception {
		int len = Measures.length;
		double[] result2 = new double[len];
		double[] times = new double[result.length-len];
		for(int i=0; i<len; i++) {
			result2[i] = result[i];
		}
		for(int i=len; i<result.length; i++) {
			times[i-len] = result[i];
		}
		prtConclusion(result2, times, dataSet, alg);
	}
	
	public double evaluating_rankingdiff(double[] output, int[] target) throws Exception {
		double max = 1;
		double min = 0;
		int cnt = 1;
		for(int i=0; i<target.length; i++) {
			if(target[i]==1) {
				max += output[i];
				cnt++;
//				if(output[i] < max) {
//					max = output[i];
//				}
			}
			else {
//				if(output[i] > min) {
//					min = output[i];
//				}
				min += output[i];
			}
		}
//		return max-min;
		return max/cnt-min/(target.length-cnt);
	}
	
	public double evaluating_rankingdiff(double[][] output, int[][] target) throws Exception {
		if((output.length != target.length) || (output[0].length != target[0].length)) {
			return -1;
		}
		double rankingdifference = 0;
		for(int i=0; i<target.length; i++) {
			rankingdifference += evaluating_rankingdiff(output[i], target[i]);
		}
		return rankingdifference/target.length;
	}
	
	public double[] evaluating(double[][] output, int[][] target) throws Exception {
		if((output.length != target.length) || (output[0].length != target[0].length)) {
			return null;
		}
		int[][] y = double2int(output);	//translate confidence value to {0,1}
		double[] result_1 = evaluatingNoSequence(y, target);	//conventional evaluation: evaluate without sequence.
		
		int[][] seq = getSeq(output);	//{0.45, 0.66, 0.73, 0.29, 0.80} ->> {4, 2, 1, 0, 3}
		int[][] rank = getRank(seq);	//{0.45, 0.66, 0.73, 0.29, 0.80} ->> {4, 3, 2, 5, 1}
		double[] result_2 = evaluatingWithSequence(y, target, seq, rank);	//sequenced evaluation
		
		return Mat.vectcat(result_1, result_2);
	}
	
	//use the authority methods
	public double[] evaluating(MultiLabelOutput[] outs, MultiLabelInstances testdata) throws Exception {
		if(outs.length != testdata.getNumInstances()) {
			return null;
		}
		int[][] y = double2int(multilabelOut2scores(outs));	//translate confidence value to {0,1}
		double[] result_1 = evaluatingNoSequence(y, Instruments.getTarget(testdata));	//conventional evaluation: evaluate without sequence.
		
//		SubsetAccuracy mSubsetAccuracy = new SubsetAccuracy();
//		HammingLoss mHammingLoss = new HammingLoss();
		OneError mOneError = new OneError();
		Coverage mCoverage = new Coverage();
		RankingLoss mRankingLoss = new RankingLoss();
		AveragePrecision mAveragePrecision = new AveragePrecision();
		boolean[][] truth = Instruments.getTruth(testdata);
		for(int i=0; i<outs.length; i++) {
//			mSubsetAccuracy.update(outs[i], new GroundTruth(truth[i]));
//			mHammingLoss.update(outs[i], new GroundTruth(truth[i]));
			mOneError.update(outs[i], new GroundTruth(truth[i]));
			mCoverage.update(outs[i], new GroundTruth(truth[i]));
			mRankingLoss.update(outs[i], new GroundTruth(truth[i]));
			mAveragePrecision.update(outs[i], new GroundTruth(truth[i]));
		}
		double[] result_2 = {mOneError.getValue(), mCoverage.getValue()/y[0].length, mRankingLoss.getValue(), mAveragePrecision.getValue()};
		return Mat.vectcat(result_1, result_2);
	}
	
	private double[] evaluatingNoSequence(int[][] y, int[][] t) throws Exception {
		int dimData = t.length;
//		double[] thisAcc = new double[dimData];
//		double[] thisHam = new double[dimData];
		double acc=0, precision=0, recall=0, f1=0, hitrate=0, subsetAcc=0, hamming=0, tmp=0;
		for(int i=0; i<dimData; i++) {
			//for each label set
			int a=0,b=0,c=0,d=0;
			for(int j=0; j<t[0].length; j++) {
				if(t[i][j]==1 && y[i][j]==1) {
					a++;	//TP
				}
				else if(t[i][j]==1 && y[i][j]==0) {
					b++;	//FN
				}
				else if(t[i][j]==0 && y[i][j]==1) {
					c++;	//FP
				}
				else if(t[i][j]==0 && y[i][j]==0) {
					d++;	//TN
				}
				else {
					System.out.println("Evaluations Error: the value of labels is nor 0 or 1.");
				}
			}
			tmp = (double)a/(a+b+c);
			
//			thisAcc[i] = acc;
			if(tmp<=1) {
				acc += tmp;
			}
			else {
				acc += 1;
			}
//			thisAcc[i] = acc - thisAcc[i];
			
			if(a>0) {
				hitrate += 1;
			}
			
//			tmp = (double)a/(a+c);
//			if(tmp<=1) {
//				precision += tmp;
//			}
//			else {
//				precision += 0;	//change: 1 to 0.
//			}
//			tmp = (double)a/(a+b);
//			if(tmp<=1) {
//				recall += tmp;
//			}
//			else {
//				recall += 1;
//			}
			if(a+b+c == 0) {
				precision += 1;
				recall += 1;
			}
			else {
				if(a+c != 0) {
					precision += (double)a/(a+c);
				}
				if(a+b != 0) {
					recall += (double)a/(a+b);
				}
			}
			
			
			
			
			tmp = (double)2*a/(2*a+b+c);
			if(tmp<=1) {
				f1 += tmp;
			}
			else {
				f1 += 1;
			}
			
//			thisHam[i] = hamming;
			hamming += (double)(b+c)/(a+b+c+d);
//			thisHam[i] = hamming - thisHam[i];
			
			if(b==0 && c==0) {
				subsetAcc +=1;
			}
		}
		acc /= dimData;
		precision /= dimData;
		recall /= dimData;
		f1 /= dimData;
		hitrate /= dimData;
		subsetAcc /= dimData;
		hamming /= dimData;
		double[] result = {acc, precision, recall, f1, hitrate, subsetAcc, hamming};	//accuracy, precision, recall, F1-measure, hitRate
		
//		Test_MLDE.setMatrix(thisAcc, "measure_Acc", Test_MLDE.isEffect);
//		Test_MLDE.setMatrix(thisHam, "measure_Ham", Test_MLDE.isEffect);
		return result;
	}
	
	private double[] evaluatingWithSequence(int[][] y, int[][] t, int[][] seq, int[][] rank) throws Exception {
		int dimTest = y.length;
		int numLabel = y[0].length;
		
		double oneerror=0, coverage=0, rankingloss=0, avg_precision=0;
		for(int i=0; i<dimTest; i++) {
//			int dim_yi = (int) Mat.sum(y[i]);
			int dim_ti = (int) Mat.sum(t[i]);
			
			if(t[i][seq[i][0]] != 1) {
				oneerror++;	//error, when the most confident one is incorrectly estimated
			}
			
			int cnt_cov = dim_ti;
			int r=0;
			for(; r<numLabel && cnt_cov!=0; r++) {
				if( t[i][seq[i][r]]==1 ) {	//all labels (in truth) should be coverage.
					cnt_cov--;
				}
			}
			coverage += r;
			if(dim_ti!=0 && dim_ti!=numLabel) {	//change: change the condition
				int cnt_rank = 0;
				for(int j=0; j<numLabel; j++) {
					if(t[i][j]==0) {
						continue;
					}
					for(int k=0; k<numLabel; k++) {
						if(t[i][k]==1) {
							continue;
						}
						if(rank[i][j] > rank[i][k]) {
							cnt_rank++;
						}
					}
				}
				rankingloss += cnt_rank/(double)(dim_ti*(numLabel-dim_ti));
//				if(!(dim_ti==0 || dim_ti==numLabel)) {
//				}
			}
			
			if(dim_ti == numLabel || dim_ti == 0) {
				avg_precision += 1;
			}
			else {
				double cnt_pre = 0;
				for(int j=0; j<numLabel; j++) {
					if(t[i][j]==0) {
						continue;
					}
					int tmp = 0;
					for(int k=0; k<numLabel; k++) {
						if(t[i][k]==0) {
							continue;
						}
						if(rank[i][j] >= rank[i][k]) {
							tmp++;
						}
					}
					cnt_pre += tmp/(double)rank[i][j];
				}
				avg_precision += cnt_pre/(double)dim_ti;
			}
//			thisPre[i] = avg_precision - thisPre[i];
		}
		
		oneerror /= dimTest;
		coverage /= dimTest;
		coverage -= 1;
		coverage /= numLabel;
		rankingloss /= dimTest;
		avg_precision /= dimTest;
		double[] result = {oneerror, coverage, rankingloss, avg_precision};
		
//		Test_MLDE.setMatrix(thisCov, "measure_Cov", Test_MLDE.isEffect);
//		Test_MLDE.setMatrix(thisRkl, "measure_Rkl", Test_MLDE.isEffect);
//		Test_MLDE.setMatrix(thisPre, "measure_Pre", Test_MLDE.isEffect);
		return result;
	}
	
	//{0.45, 0.66, 0.73, 0.29, 0.80} ->> {4, 2, 1, 0, 3}
	private int[][] getSeq(double[][] y) {
		int[][] seq = new int[y.length][y[0].length];
		for(int i=0; i<y.length; i++) {
			seq[i] = arraysort(y[i]);
		}
		return seq;
	}
	
	//{0.45, 0.66, 0.73, 0.29, 0.80} ->> {4, 3, 2, 5, 1}
	private int[][] getRank(int[][] seq) {
		int[][] rank = new int[seq.length][seq[0].length];
		for(int i=0; i<seq.length; i++) {
			for(int j=0; j<seq[0].length; j++) {
				rank[i][seq[i][j]] = j+1;
			}
		}
		return rank;
	}
	
	private int[] arraysort(double[] org_arr) {
		double temp;
		int index;
		double[] arr = new double[org_arr.length];
		for (int i = 0; i < arr.length; i++) {
			arr[i] = org_arr[i];
		}
		int[] Index = new int[arr.length];
		for (int i = 0; i < arr.length; i++) {
			Index[i] = i;
		}
		for (int i = 0; i < arr.length; i++) {
			for (int j = 0; j < arr.length-i-1; j++) {
				if (arr[j] < arr[j + 1]) {
					temp = arr[j];
					arr[j] = arr[j + 1];
					arr[j + 1] = temp;
					
					index = Index[j];
					Index[j] = Index[j + 1];
					Index[j + 1] = index;
				}
			}
		}
		return Index;
	}
	
	private double[][] multilabelOut2scores(MultiLabelOutput[] outs) throws Exception {
		double[][] scores = new double[outs.length][];
		for(int i=0; i<outs.length; i++) {
			scores[i] = outs[i].getConfidences();
		}
		return scores;
	}
	
	private int[][] double2int(double[][] output){
		int[][] y = new int[output.length][output[0].length];
		for(int i=0; i<output.length; i++) {
			for(int j=0; j<output[0].length; j++) {
				if(output[i][j] >= threshold) {
					y[i][j] = 1;
				}
				else {
					y[i][j] = 0;
				}
			}
		}
		return y;
	}
	
	public static void main(String[] str) throws Exception {
//		double[][] y = {{0.7, 0.8, 0.6, 0.4, 0.9}, {0.42, 0.3, 0.8, 0.7, 0.6}};
//		int[][] t_p = {{1, 1, 1, 0, 1}, {0, 0, 1, 1, 1}};
//		int[][] t_np = {{0, 0, 0, 1, 0}, {1, 1, 0, 0, 0}};
//		int[][] t1 = {{1, 0, 1, 1, 0}, {1, 0, 1, 1, 0}};
//		e.prtConclusion(e.evaluating(y, t_p));
//		e.prtConclusion(e.evaluating(y, t_np));
//		e.prtConclusion(e.evaluating(y, t1));
		
		System.out.println("ALL BEGIN: Test_Ensemble");
		String[] filename = {"3sources_bbc1000","3sources_guardian1000","3sources_inter3000","3sources_reuters1000","Birds","CAL500","CHD_49","Emotions","Flags","Foodtruck",
				"GnegativePseAAC","GpositiveGO","GpositivePseAAC","Image","PlantPseAAC","Scene","VirusGO","VirusPseAAC","Water_quality","Yeast"};
		for(int fileIndex = 0; fileIndex < 1; fileIndex++) {
			String dataFileName = filename[fileIndex];
			dataFileName = "Emotions";
			
			String path = "E:\\multiLabel\\code_cui\\mulan-cuiwei\\mulan\\datatest\\ALLdata\\DATA2GEN\\";
			path += dataFileName;
			path += "\\";
			
			String arffFile_train = path + dataFileName + "-train.arff";
			String arffFile_test = path + dataFileName + "-test.arff";
			String xmlFile = path + dataFileName + ".xml";
			MultiLabelInstances origData_train = new MultiLabelInstances(arffFile_train, xmlFile);
			MultiLabelInstances origData_test = new MultiLabelInstances(arffFile_test, xmlFile);
			
			MLkNN mLearner = new MLkNN();
			mLearner.build(origData_train);
			
			//FULL authority method to calculate conclusion
			List<Measure> measures = new ArrayList<>(10);
			measures.add(new ExampleBasedAccuracy());
			measures.add(new ExampleBasedPrecision());
			measures.add(new ExampleBasedRecall());
			measures.add(new ExampleBasedFMeasure());
			measures.add(new SubsetAccuracy());
			measures.add(new HammingLoss());
			measures.add(new OneError());
			measures.add(new Coverage());
			measures.add(new RankingLoss());
			measures.add(new AveragePrecision());
			Evaluator mEvaluator = new Evaluator();
			Evaluation mEvaluation = mEvaluator.evaluate(mLearner, origData_test, measures);
			System.out.println(mEvaluation);
			
			//SEMI authority method to calculate conclusion
//			MultiLabelOutput[] outputs = new MultiLabelOutput[origData_test.getNumInstances()];
//			Instances mInstances = origData_test.getDataSet();
//			for(int i=0; i<origData_test.getNumInstances(); i++) {
//				outputs[i] = mLearner.makePrediction(mInstances.get(i));
//			}
//			e.prtConclusion(e.evaluating(outputs, origData_test));
			
		}
		
		System.out.println("finished.");
	}
}
