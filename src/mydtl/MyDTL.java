package mydtl;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class MyDTL {
    private static final String DATA_SOURCE = "weather.nominal.arff";
    private static final String DATA_SOURCE_UNLABELED = "weather.nominal.unlabeled.arff";
    
    public static void main(String[] args) {
        
        // Load file from data source
        Instances instances = Helper.loadDataFromFile(DATA_SOURCE);

        Classifier classifier_myid3 = Helper.buildClassifier(instances, "myid3");
        System.out.println(classifier_myid3.toString());
      
        System.out.println("============================================");
        Helper.classifyUsingModel(classifier_myid3, DATA_SOURCE_UNLABELED);
        System.out.println("============================================");
    }
}
