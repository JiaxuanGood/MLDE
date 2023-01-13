package myStack;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.EuclideanDistance;
import weka.core.Instances;

public class MultiInstanceHandler {
	public static Instances getInstanceNoLabels(MultiLabelInstances mulData) {
		int[] labels = mulData.getLabelIndices();
		Instances data = new Instances(mulData.getDataSet());
		for(int i=labels.length-1; i>=0; i--) {
			data.deleteAttributeAt(i);
		}
		return data;
	}
	
	public static MultiLabelInstances fixMultiData(MultiLabelInstances origData, int[] wantIndex) throws InvalidDataFormatException {
		Instances data = origData.getDataSet();
		Instances wantData = new Instances(data, 0);
		for(int i=0; i<wantIndex.length; i++) {
			wantData.add(data.instance(wantIndex[i]));
		}
		return new MultiLabelInstances(wantData, origData.getLabelsMetaData());
	}
	
	public static int[] getIndexWant(Instances orgData, Instances purnData) {
		int capOrg = orgData.numInstances();
		int capWant = purnData.numInstances();
		int[] wantIndex = new int[capWant];
		EuclideanDistance L2Distance = new EuclideanDistance();
		L2Distance.setInstances(orgData);
		for(int i=0; i<capWant; i++) {
			for(int j=0; j<capOrg; j++) {
				double thisdistance = L2Distance.distance(purnData.instance(i), orgData.instance(j));
				if(thisdistance==0.0) {
					wantIndex[i] = j;
				}
			}
		}
		return wantIndex;
	}
	
	public static int[] getIndexWant(MultiLabelInstances orgMulData, Instances purnData) {
		Instances orgData = getInstanceNoLabels(orgMulData);
		int capOrg = orgData.numInstances();
		int capWant = purnData.numInstances();
		int[] wantIndex = new int[capWant];
		EuclideanDistance L2Distance = new EuclideanDistance();
		L2Distance.setInstances(orgData);
		for(int i=0; i<capWant; i++) {
			for(int j=0; j<capOrg; j++) {
				double thisdistance = L2Distance.distance(purnData.instance(i), orgData.instance(j));
				if(thisdistance==0.0) {
					wantIndex[i] = j;
				}
			}
		}
		return wantIndex;
	}
}
