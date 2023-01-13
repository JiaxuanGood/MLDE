package myStack;

import java.io.File;
import mulan.data.MultiLabelInstances;
import myBaseLearners.BR;

public class Test_MLDE_Ablation {
	//hyper-parameters
	static int L = 28;	//num of base learners
	static int numKind = 7;
	static int numEach = 4;
	static int numBrothers = 10;	//capacity of competence region
	static int distMetricFlag = 2;	//Euclidean distance
	//global variable
	static boolean isSave = false;
	static String dataFileName = "null";
	static int numLabel = 0;
	static int dimTrain = 0;
	static int dimTest = 0;
	
	public static void dynamicEnsemble(MultiLabelInstances[] basesMul, MultiLabelInstances dataMul_test, MultiLabelInstances[] validsMul) throws Exception {
		long t0 = System.currentTimeMillis();
		
		//data prepare: convert original MultiLabelInstances to single Instances
		EvaluationsMul evaluations = new EvaluationsMul();
		Evaluations evaluation = new Evaluations();
		int numGood = L/2;
		
		BR[] mBrs = new BR[L];
		for(int i=0; i<L; i++) {
			if(isSave) {
				mBrs[i] = new BR(dataFileName, i, numLabel);
			}
			else {
				mBrs[i] = new BR(i, basesMul[i]);
			}
		}
//		MultiBRThreader mMultiBRThreader = new MultiBRThreader();
//		BR[] mBrs = mMultiBRThreader.trainThreader(basesMul);
		
		int dimValid = validsMul.length;
		System.out.println(dimValid);
		int[][][] indexs_slacc = new int[dimValid][numLabel][numGood];
		int[][][] indexs_slacc2 = new int[dimValid][numLabel][numGood];
		int[][] indexs_mlacc = new int[dimValid][numGood];
		int[][] indexs_ranking = new int[dimValid][numGood];
		for(int j=0; j<dimValid; j++) {
			double[] ml_accuracy = new double[L];
			double[] rankingloss = new double[L];
			double[][] sl_accuracyes = new double[L][numLabel];
			for(int i=0; i<L; i++) {
				double[][] y_v = mBrs[i].predict(validsMul[j]);
				int[][] t_v = Instruments.getTarget(validsMul[j]);
				double[] e = evaluations.evaluating(y_v, t_v);
				ml_accuracy[i] = -e[6];
				rankingloss[i] = -e[9];
				sl_accuracyes[i] = evaluation.evaluateSingle_Accs(y_v, t_v);
			}
			indexs_mlacc[j] = Instruments.getTopM3(ml_accuracy, numGood);
			indexs_ranking[j] = Instruments.getTopM3(rankingloss, numGood);
			
			sl_accuracyes = Mat.T(sl_accuracyes);
			for(int k=0; k<numLabel; k++) {
				indexs_slacc[j][k] = Instruments.getsort(sl_accuracyes[k]);
				indexs_slacc2[j][k] = Instruments.getTopM3(sl_accuracyes[k], numGood);
			}
		}
		
		double[][] predict_ranking = Run_MLDE.predicting_metric(mBrs, dataMul_test, indexs_ranking);
		double[][] predict_mlacc = predicting_metric(mBrs, dataMul_test, indexs_mlacc);
		double[][] predict_static = predicting_nometric(mBrs, dataMul_test);
		double[][] predict_slacc = Run_MLDE.predicting_metrics(mBrs, dataMul_test, indexs_slacc);
		double[][] predict_slacc2 = predicting_metrics2(mBrs, dataMul_test, indexs_slacc2);
		
		int[][] t_test = Instruments.getTarget(dataMul_test);
		evaluations.prtConclusion(evaluations.evaluating(predict_static, t_test), (System.currentTimeMillis()-t0), dataFileName, "MLDE_static");
		evaluations.prtConclusion(evaluations.evaluating(predict_slacc, t_test), (System.currentTimeMillis()-t0), dataFileName, "MLDE_slacc");
		evaluations.prtConclusion(evaluations.evaluating(predict_ranking, t_test), (System.currentTimeMillis()-t0), dataFileName, "MLDE_ranking");
		evaluations.prtConclusion(evaluations.evaluating(Run_MLDE.mixing(predict_mlacc, predict_ranking), t_test), (System.currentTimeMillis()-t0), dataFileName, "MLDE_mlacc_ranking");
		evaluations.prtConclusion(evaluations.evaluating(Run_MLDE.mixing(predict_slacc, predict_ranking), t_test), (System.currentTimeMillis()-t0), dataFileName, "MLDE_slacc_ranking");
		evaluations.prtConclusion(evaluations.evaluating(Run_MLDE.mixing(predict_slacc2, predict_ranking), t_test), (System.currentTimeMillis()-t0), dataFileName, "MLDE_slacc2_ranking");
	}
	
	private static double[][] predicting_metrics2(BR[] mBrs, MultiLabelInstances dataMul_test, int[][][] metricIndex) throws Exception{
		double[][] predict_this = new double[dimTest][numLabel];
		int numGood = L/2;
		for(int j=0; j<dimTest; j++) {
			MultiLabelInstances thisTest = Run_MLDE.splitInstances(dataMul_test, j);
			for(int k=0; k<numLabel; k++) {
				for(int r=0; r<numGood; r++) {
					predict_this[j][k] += mBrs[metricIndex[j][k][r]].predict_prob(thisTest, k) [0];
				}
				predict_this[j][k] /= numGood;
			}
		}
		return predict_this;
	}
	
	private static double[][] predicting_metric(BR[] mBrs, MultiLabelInstances dataMul_test, int[][] metricIndex) throws Exception{
		int numGood = L/2;
		double[][] predict_this = new double[dimTest][numLabel];
		for(int j=0; j<dimTest; j++) {
			int index_valid = j;
			MultiLabelInstances thisTest = Run_MLDE.splitInstances(dataMul_test, j);
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
	
	private static double[][] predicting_nometric(BR[] mBrs, MultiLabelInstances dataMul_test) throws Exception{
		double[][] predict_this = new double[dimTest][numLabel];
		for(int r=0; r<L; r++) {
			predict_this = Mat.add(predict_this, mBrs[r].predict(dataMul_test));
		}
		predict_this = Mat.dot(predict_this, 1/L);
		return predict_this;
	}
	
	public static void main(String[] args) throws Exception {
		System.out.println("ALL BEGIN: Test_MLDE_Ablation");
		
		File directory = new File("");
		String dirpath = directory.getAbsolutePath();
		
		//down load benchmark data sets from KIDS: http://www.uco.es/kdis/mllresources/
		String[] filename = {"3sources_bbc1000","3sources_guardian1000","3sources_inter3000","3sources_reuters1000","Birds","CAL500","CHD_49","Emotions",
			    "Flags","Foodtruck","Genbase","GnegativeGO","GpositiveGO","HumanGO","Image","Medical",
			    "PlantGO","Scene","Coffee","tmc2007_500","VirusGO","Water_quality","Yeast","Yelp"};
//		String[] filename = {"Flags","Yeast","Medical","CAL500"};	//4 representative datasets
		
		L = 28;
		numEach = 4;
		numKind = 7;
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
				System.out.println(dataFileName + ": " + dimTrain + ":" + dimTest + ":" + numLabel);
				
				MultiLabelInstances[] validDatas = Run_MLDE.findNeighbors(origData_train, origData_test, numBrothers, distMetricFlag);	//an instance in test data <==> a valid set
				MultiLabelInstances[] basedatas = Run_MLDE.bootStrap(origData_train);
				dynamicEnsemble(basedatas, origData_test, validDatas);
			}
		}
		System.out.println("ALL FINISHED");
	}
}
