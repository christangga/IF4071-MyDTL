package mydtl;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class MyDTL {

    private static final String DATA_SOURCE = "data/iris.arff";
    private static final String DATA_SOURCE_UNLABELED = "data/iris.unlabeled.arff";

//    public static void main(String[] args) {
//
//        // Load file from data source
//        Instances instances = Helper.loadDataFromFile(DATA_SOURCE);
//
//        System.out.println("=================================================");
//        Classifier classifier_id3 = Helper.buildClassifier(instances, "j48");
//        System.out.println(classifier_id3.toString());
//        System.out.println("=================================================");
//        Classifier classifier_myid3 = Helper.buildClassifier(instances, "myj48");
//        System.out.println(classifier_myid3.toString());
//        System.out.println("=================================================\n");
//
//        System.out.println("==============================");
//        Helper.classifyUsingModel(classifier_id3, DATA_SOURCE_UNLABELED);
//        System.out.println("==============================");
//        Helper.classifyUsingModel(classifier_myid3, DATA_SOURCE_UNLABELED);
//        System.out.println("===============================");
//    }
}
