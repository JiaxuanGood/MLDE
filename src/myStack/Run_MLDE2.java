package myStack;

import java.io.File;
import java.util.Random;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import myBaseLearners.BR;
import myBaseLearners.SingleLearners;
import weka.core.DistanceFunction;
import weka.core.Instances;
import weka.core.neighboursearch.LinearNNSearch;

public class Run_MLDE2 {
	//hyper-parameters
	static int L = 20;	//num of base learners
	static int numKind = 4;
	static int numEach = 5;
	static int numBrothers = 5;	//capacity of competence region
	static int distMetricFlag = 2;	//Euclidean distance
	//global variable
	static String dataFileName = "null";
	static int numLabel = 0;
	static int dimTrain = 0;
	static int dimTest = 0;
	public static boolean isMultiThread = true;
	
	public static void dynamicEnsemble(MultiLabelInstances[] basesMul, MultiLabelInstances dataMul_train, MultiLabelInstances dataMul_test, int[][] pre_id) throws Exception {
		long t0 = System.currentTimeMillis();
		
		//data prepare: convert original MultiLabelInstances to single Instances
		EvaluationsMul evaluations = new EvaluationsMul();
		Evaluations evaluation = new Evaluations();
		int numGood = L/2;
		
		BR[] mBrs = new BR[L];
		for(int i=0; i<L; i++) {
			if(BR.isExist(dataFileName) && basesMul==null) {
				mBrs[i] = new BR(dataFileName, i, numLabel);
			}
			else {	//train BR in multiple thread
				// 1) training single-label classifier in each base BR classifier synchronously
				mBrs[i] = new BR(i, basesMul[i]);	
				// 2) training base multi-label BR classifiers synchronously
//				MultiBRThreader mMultiBRThreader = new MultiBRThreader();
//				mBrs = mMultiBRThreader.trainThreader(basesMul);
			}
		}
		
		int[][] pre_target = Instruments.getTarget(dataMul_train);
		double[][][] pre_prediction = new double[L][][];
		for (int i=0; i<L; i++) {
			pre_prediction[i] = mBrs[i].predict(dataMul_train);
		}
		
		long t1 = System.currentTimeMillis();
		int dimValid = pre_id.length;
		int[][][] indexs_slacc = new int[dimValid][numLabel][numGood];
		int[][] indexs_ranking = new int[dimValid][numGood];
		
		for(int j=0; j<dimValid; j++) {
			double[] rankingloss = new double[L];
			double[][] sl_accuracyes = new double[L][numLabel];
			for(int i=0; i<L; i++) {
				double[][] y_v = Mat.getwant(pre_prediction[i], pre_id[j]);
				int[][] t_v = Mat.getwant(pre_target, pre_id[j]);
				double[] e = evaluations.evaluating(y_v, t_v);
				rankingloss[i] = -e[9];
				sl_accuracyes[i] = evaluation.evaluateSingle_Accs(y_v, t_v);
			}
			indexs_ranking[j] = Instruments.getTopM3(rankingloss, numGood);
			
			sl_accuracyes = Mat.T(sl_accuracyes);
			for(int k=0; k<numLabel; k++) {
				indexs_slacc[j][k] = Instruments.getsort(sl_accuracyes[k]);
//				Mat.equal(Instruments.sort(sl_accuracyes[k]), Instruments.getsort(sl_accuracyes[k]));
			}
		}
		
		long t2 = System.currentTimeMillis();
		
		
		double[][] predict_ranking = predicting_metric(mBrs, dataMul_test, indexs_ranking);
		double[][] predict_slacc = predicting_metrics(mBrs, dataMul_test, indexs_slacc);
		double[][] predict_last = mixing(predict_slacc, predict_ranking);
		
//		double[][] predict_last2 = predicting_metric_para(mBrs, dataMul_test, indexs_ranking, indexs_slacc);
//		Mat.equal(predict_last, predict_last2);
		
		int[][] t_test = Instruments.getTarget(dataMul_test);
		double[] alltimes = {t1-t0, t2-t1, System.currentTimeMillis()-t2};
		evaluations.prtConclusion(evaluations.evaluating(predict_last, t_test), alltimes, dataFileName, "MLDE");
	}
	
	public static double[][] predicting_metrics(BR[] mBrs, MultiLabelInstances dataMul_test, int[][][] metricIndexs) throws Exception{
		double[][] predict_this = new double[dimTest][numLabel];
		for(int j=0; j<dimTest; j++) {
			MultiLabelInstances thisTest = splitInstances(dataMul_test, j);
			for(int k=0; k<numLabel; k++) {
				int numGood = metricIndexs[j][k][L];
				if(isMultiThread) {
					BR[] selectBRs = new BR[numGood];
					for(int r=0; r<numGood; r++) {
						selectBRs[r] = mBrs[metricIndexs[j][k][r]];
					}
					predict_this[j][k] = new BR().predict_prob_mulThread(selectBRs, thisTest, k);
				}else {
					for(int r=0; r<numGood; r++) {
						predict_this[j][k] += mBrs[metricIndexs[j][k][r]].predict_prob(thisTest, k) [0];
					}
					predict_this[j][k] /= numGood;
				}
			}
		}
		return predict_this;
	}
	
	public static double[][] predicting_metric(BR[] mBrs, MultiLabelInstances dataMul_test, int[][] metricIndex) throws Exception{
		int numGood = L/2;
		double[][] predict_this = new double[dimTest][numLabel];
		for(int j=0; j<dimTest; j++) {
			int index_valid = j;
			MultiLabelInstances thisTest = splitInstances(dataMul_test, j);
			for(int r=0; r<numGood; r++) {
				double[] predict_ensemble = mBrs[metricIndex[index_valid][r]].predict(thisTest) [0];
				for(int z=0; z<numLabel; z++) {
					predict_this[j][z] += predict_ensemble[z];
				}
			}
			for(int z=0; z<numLabel; z++) {
				predict_this[j][z] /= numGood;
			}
		}
		return predict_this;
	}
	
	public static double[][] predicting_metric_para(BR[] mBrs, MultiLabelInstances dataMul_test, int[][] metricIndex, int[][][] metricIndexs) throws Exception{
		double[][] predict_this = new double[dimTest][numLabel];
		for(int j=0; j<dimTest; j++) {
			MultiLabelInstances thisTest = splitInstances(dataMul_test, j);
			
			predict_this[j] = predicting_metric_para(mBrs, thisTest, metricIndex[j], metricIndexs[j]);
		}
		return predict_this;
	}
	
	private static double[] predicting_metric_para(BR[] mBrs, MultiLabelInstances thisTest, int[] metricIndex, int[][] metricIndexs) throws Exception{
		double[] predict_this = new double[numLabel];
		
		int numGood = L/2;
		double[] predict_this1 = new double[numLabel];
		for(int r=0; r<numGood; r++) {
			double[] predict_ensemble = mBrs[metricIndex[r]].predict(thisTest) [0];
			for(int z=0; z<numLabel; z++) {
				predict_this1[z] += predict_ensemble[z];
			}
		}
		for(int z=0; z<numLabel; z++) {
			predict_this1[z] /= numGood;
		}
		
		for(int k=0; k<numLabel; k++) {
			int numGood2 = metricIndexs[k][L];
			double tmp2 = 0;
			for(int r=0; r<numGood2; r++) {
				tmp2 += mBrs[metricIndexs[k][r]].predict_prob(thisTest, k) [0];
			}
			tmp2 /= numGood2;
			if(tmp2 >= 0.5) {
				predict_this[k] = predict_this1[k] + 0.5;
			}
			else {
				predict_this[k] = predict_this1[k] - 0.5;
			}
		}
		
		return predict_this;
	}
	
	public static double[][] mixing(double[][] predict, double[][] predict_ranking) {
		double[][] predict_this = Mat.matrixDup2(predict_ranking);
		for(int i=0; i<dimTest; i++) {
			for(int j=0; j<numLabel; j++) {
				if(predict[i][j] >= 0.5) {
					predict_this[i][j] += 0.5;
				}
				else {
					predict_this[i][j] -= 0.5;
				}
			}
		}
		return predict_this;
	}
	
	public static MultiLabelInstances[] bootStrap(MultiLabelInstances orgData) throws Exception {
		MultiLabelInstances[] multiLabelInstanceses = new MultiLabelInstances[L];
		for(int i=0; i<L; i++) {
			int thisCap = orgData.getNumInstances();
			multiLabelInstanceses[i] = resample(orgData, thisCap);
		}
		return multiLabelInstanceses;
	}
	
	public static MultiLabelInstances resample(MultiLabelInstances orgData, int cap) throws Exception {
		Instances orgInstances = orgData.getDataSet();
		Instances thisBase = new Instances(orgInstances, 0);
		Random r = new Random();
		for(int j=0; j<cap; j++) {
			int thisIndex = r.nextInt(cap);
			thisBase.add(orgInstances.instance(thisIndex));
		}
		return new MultiLabelInstances(thisBase, orgData.getLabelsMetaData());
	}
	
	public static MultiLabelInstances[] findNeighbors(MultiLabelInstances trainInstances, MultiLabelInstances newInstance, int numOfBrothers, int distFlag) throws Exception {
		Instances[] knnInstances = new Instances[newInstance.getNumInstances()];
		MultiLabelInstances[] knnMultiLabelInstances = new MultiLabelInstances[newInstance.getNumInstances()];
		Instances train = MultiInstanceHandler.getInstanceNoLabels(trainInstances);
		Instances newOne = MultiInstanceHandler.getInstanceNoLabels(newInstance);
		
		LinearNNSearch linearnn = new LinearNNSearch();
		
		DistanceFunction dFunction = SingleLearners.getDistFunction(distFlag);
		linearnn.setDistanceFunction(dFunction);
		linearnn.setInstances(train);
		
		double count = 0;
		for(int i=0; i<newInstance.getNumInstances(); i++) {
			knnInstances[i] = linearnn.kNearestNeighbours(newOne.instance(i), numOfBrothers);
			count += knnInstances[i].numInstances();
			int[] wantIndex = MultiInstanceHandler.getIndexWant(train, knnInstances[i]);
			knnMultiLabelInstances[i] = MultiInstanceHandler.fixMultiData(trainInstances, wantIndex);
		}
		System.out.println("avg_numBro\t" + count/newInstance.getNumInstances());
		return knnMultiLabelInstances;
	}
	
	public static int[][] findNeighborsId(MultiLabelInstances trainInstances, MultiLabelInstances newInstance, int numOfBrothers, int distFlag) throws Exception {
		Instances[] knnInstances = new Instances[newInstance.getNumInstances()];
		Instances train = MultiInstanceHandler.getInstanceNoLabels(trainInstances);
		Instances newOne = MultiInstanceHandler.getInstanceNoLabels(newInstance);
		
		LinearNNSearch linearnn = new LinearNNSearch();
		DistanceFunction dFunction = SingleLearners.getDistFunction(distFlag);
		linearnn.setDistanceFunction(dFunction);
		linearnn.setInstances(train);
		
		int[][] nerghborsId = new int[newInstance.getNumInstances()][numOfBrothers];
		for(int i=0; i<newInstance.getNumInstances(); i++) {
			knnInstances[i] = linearnn.kNearestNeighbours(newOne.instance(i), numOfBrothers);
			nerghborsId[i] = MultiInstanceHandler.getIndexWant(train, knnInstances[i]);
		}
		return nerghborsId;
	}
	
	public static MultiLabelInstances splitInstances(MultiLabelInstances mulInst, int i) throws InvalidDataFormatException {
		Instances orgInstances = mulInst.getDataSet();
		Instances mInstances = new Instances(orgInstances, 0);
		mInstances.add(orgInstances.instance(i));
		MultiLabelInstances mMultiLabelInstances = new MultiLabelInstances(mInstances, mulInst.getLabelsMetaData());
		return mMultiLabelInstances;
	}
	
	public static void main(String[] args) throws Exception {
		System.out.println("ALL BEGIN: Run_MLDE");
		
		File directory = new File("");
		String dirpath = directory.getAbsolutePath();
		
		//down load benchmark data sets from KIDS: http://www.uco.es/kdis/mllresources/
		String[] filename = {"3sources_bbc1000","3sources_guardian1000","3sources_inter3000","3sources_reuters1000","Birds","CAL500","CHD_49","Emotions",
			    "Flags","Foodtruck","Genbase","GnegativeGO","GpositiveGO","HumanGO","Image","Medical",
			    "PlantGO","Scene","Coffee","tmc2007_500","VirusGO","Water_quality","Yeast","Yelp"};
//		String[] filename = {"Flags","Yeast","Medical","CAL500"};	//4 representative datasets
		
		L = 20;
		numEach = 5;
		numKind = 4;
		SingleLearners.numKind = numKind;
		numBrothers = 5;
		for(int testtimes=0; testtimes<1; testtimes++) {
			for(int fileIndex = 0; fileIndex < 24; fileIndex++) {
				dataFileName = filename[fileIndex];
				
				String path = dirpath + "\\data\\" + dataFileName + "\\";
				String arffFile_train = path + dataFileName + "-train.arff";
				String arffFile_test = path + dataFileName + "-test.arff";
				String xmlFile = path + dataFileName + ".xml";
				MultiLabelInstances origData_train = new MultiLabelInstances(arffFile_train, xmlFile);
				MultiLabelInstances origData_test = new MultiLabelInstances(arffFile_test, xmlFile);
				
				dimTrain = origData_train.getNumInstances();
				dimTest = origData_test.getNumInstances();
				numLabel = origData_train.getNumLabels();
//				origData_train.getDataSet().attribute(0).isNominal();
				System.out.println(dataFileName + ": " + dimTrain + ":" + dimTest + ":" + numLabel);
				
				int[][] validDatas_id = findNeighborsId(origData_train, origData_test, numBrothers, distMetricFlag);	//an instance in test data <==> a valid set
				
				MultiLabelInstances[] basedatas = bootStrap(origData_train);
				dynamicEnsemble(null, origData_train, origData_test, validDatas_id);
			}
		}
		System.out.println("ALL FINISHED");
	}
}
