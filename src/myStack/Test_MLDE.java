package myStack;

import java.io.File;
import mulan.data.MultiLabelInstances;
import myBaseLearners.BR;
import myBaseLearners.SingleLearners;

/*
 * Author: Jiaxuan Li
 * Perform training and testing separately to record their time. The induced model will be saved. 
 * */

public class Test_MLDE {
	//hyper-parameters
	static int L = 20;	//num of base learners
	static int numKind = 4;
	static int numEach = 5;
	static int numBrothers = 5;	//capacity of competence region
	
	static String dataFileName = "null";
	static int numLabel = 0;
	static int dimTrain = 0;
	static int dimTest = 0;
	
	public static void main(String[] args) throws Exception {
		System.out.println("ALL BEGIN: Test_MLDE_comparing");
		
		File directory = new File("");
		String dirpath = directory.getAbsolutePath();
		String[] filename = {"3sources_bbc1000","3sources_guardian1000","3sources_inter3000","3sources_reuters1000","Birds","CAL500","CHD_49","Emotions",
		    "Flags","Foodtruck","Genbase","GnegativeGO","GpositiveGO","HumanGO","Image","Medical",
		    "PlantGO","Scene","Coffee","tmc2007_500","VirusGO","Water_quality","Yeast","Yelp"};
//		String[] filename = {"Flags","Yeast","Medical","CAL500"};	//4 representative ones
		L = 20;
		numEach = 5;
		numKind = 4;
		SingleLearners.numKind = numKind;
		numBrothers = 5;
		Run_MLDE.L = L;
		Run_MLDE.numEach = numEach;
		Run_MLDE.numKind = numKind;
		Run_MLDE.numBrothers = numBrothers;
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
			Run_MLDE.dimTrain = dimTrain;
			Run_MLDE.dimTest = dimTest;
			Run_MLDE.numLabel = numLabel;
			Run_MLDE.dataFileName = dataFileName;
			System.out.println(dataFileName + ": " + dimTrain + ":" + dimTest + ":" + numLabel);
			
			Run_MLDE.isMultiThread = false;
			
			double[] traintime = new double[L+1];
			double[] trainmem = new double[L];
			double[] testtime = new double[1];
			if(!BR.isExist(dataFileName)) {
				long t0 = System.currentTimeMillis();
				System.out.println("Train and save only! Run again after this finish.");
				MultiLabelInstances[] basedatas = Run_MLDE.bootStrap(origData_train);
				for(int i=0; i<L; i++) {
					System.out.println("Train the " + i + "-th learner.");
					double mem = Instruments.freeMem();
					long t = System.currentTimeMillis();
					new BR(i, basedatas[i], dataFileName);	//induce a BR model, false for non-multiple thread.
					traintime[i] = System.currentTimeMillis() - t;
					trainmem[i] = mem-Instruments.freeMem();
				}
				traintime[L] = System.currentTimeMillis() - t0;
				Instruments.setMatrix(traintime, "traintime");
				Instruments.setMatrix(trainmem, "trainmem");
			}
			else {
				long t = System.currentTimeMillis();
				MultiLabelInstances[] validDatas = Run_MLDE.findNeighbors(origData_train, origData_test, numBrothers, 2);	//an instance in test data <==> a valid set
				System.out.println("No train! Read the saved model for classification.");
				Run_MLDE.dynamicEnsemble(null, origData_test, validDatas);
				testtime[0] = System.currentTimeMillis() - t;
				Instruments.setMatrix(testtime, "testtime");
			}
		}
		System.out.println("ALL FINISHED");
	}
}
