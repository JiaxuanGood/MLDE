//package myStack;
//
//import mulan.data.InvalidDataFormatException;
//import mulan.data.MultiLabelInstances;
//import weka.core.Instance;
//import weka.core.Instances;
//
///*
// * author: JiaxuanLi
// * */
//
//public class InstancesSplition {
//	public static MultiLabelInstances getSome(MultiLabelInstances mulInst, int cap) throws InvalidDataFormatException {
//		Instances mInstances = new Instances(mulInst.getDataSet());
//		Instances rInstances = new Instances(mInstances, 0);
//		for(int i=0; i<cap; i++) {
//			rInstances.add(mInstances.instance(i));
//		}
//		return new MultiLabelInstances(rInstances, mulInst.getLabelsMetaData());
//	}
//	
//	public static Instances getSome(Instances inst, int[] index) throws InvalidDataFormatException {
//		Instances mInstances = new Instances(inst, 0);
//		for(int i=0; i<index.length; i++) {
//			Instance wantInstance = inst.instance(index[i]);
//			mInstances.add(wantInstance);
//		}
//		return mInstances;
//	}
//	
//	public static Instances getOne(Instances inst, int i) throws InvalidDataFormatException {
//		Instance wantInstance = inst.instance(i);
//		Instances mInstances = new Instances(inst, 0);
//		mInstances.add(wantInstance);
//		return mInstances;
//	}
//	
//	
//}
