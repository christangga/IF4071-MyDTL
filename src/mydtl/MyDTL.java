package mydtl;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class MyDTL {
    private static final String DATA_SOURCE = "weather.nominal.arff";
    private static final String DATA_SOURCE_UNLABELED = "weather.nominal.unlabeled.arff";
    
    public static void main(String[] args) {
        
        // Load file from data source
        Instances instances = Helper.loadDataFromFile(DATA_SOURCE);
        
        // Remove the first attribute which is outlook
        // instances = Helper.removeAttribute(instances, "1");
        
        // Resample the dataset
        // instances = Helper.resample(instances);
        
        // Try to build classifiers using naive bayes, id3, & j48
        Classifier classifier_naivebayes = Helper.buildClassifier(instances, "naivebayes");
        Classifier classifier_id3 = Helper.buildClassifier(instances, "id3");
        Classifier classifier_myid3 = Helper.buildClassifier(instances, "myid3");
        Classifier classifier_j48 = Helper.buildClassifier(instances, "j48");
        Classifier classifier_myj48 = Helper.buildClassifier(instances, "myj48");
        System.out.println(classifier_naivebayes.toString());
        System.out.println("================================================");
        System.out.println(classifier_id3.toString());
        System.out.println("================================================");
        System.out.println(classifier_myid3.toString());
        System.out.println("================================================");
        System.out.println(classifier_j48.toString());
        System.out.println("================================================");
        System.out.println(classifier_myj48.toString());
        
        // Do ten fold cross validation using the 3 classifiers
//        System.out.println("*********************************************************************");
//        System.out.println("*                                                                   *");
//        System.out.println("*                      TEN FOLD CROSS VALIDATION                    *");
//        System.out.println("*                                                                   *");
//        System.out.println("*********************************************************************");
//        Helper.tenFoldCrossValidation(instances, classifier_naivebayes);
//        System.out.println("================================================");
//        Helper.tenFoldCrossValidation(instances, classifier_id3);
//        System.out.println("================================================");
//        Helper.tenFoldCrossValidation(instances, classifier_j48);
//
//        // Do 80% percentage split to test the data
//        System.out.println("*********************************************************************");
//        System.out.println("*                                                                   *");
//        System.out.println("*                           PERCENTAGE SPLIT                        *");
//        System.out.println("*                                                                   *");
//        System.out.println("*********************************************************************");
//        Helper.percentageSplit(instances, classifier_naivebayes, 80);
//        System.out.println("================================================");
//        Helper.percentageSplit(instances, classifier_id3, 80);
//        System.out.println("================================================");
//        Helper.percentageSplit(instances, classifier_j48, 80);
//        
//        // Try to save and load model from file
//        Helper.saveModelToFile(classifier_naivebayes, "model_naivebayes.model");
//        Helper.saveModelToFile(classifier_id3, "model_id3.model");
//        Helper.saveModelToFile(classifier_j48, "model_j48.model");
//        Classifier loadmodel_naivebayes = Helper.loadModelFromFile("model_naivebayes.model");
//        Classifier loadmodel_id3 = Helper.loadModelFromFile("model_id3.model");
//        Classifier loadmodel_j48 = Helper.loadModelFromFile("model_j48.model");
//        System.out.println(loadmodel_naivebayes.toString());
//        System.out.println("================================================");
//        System.out.println(loadmodel_id3.toString());
//        System.out.println("================================================");
//        System.out.println(loadmodel_j48.toString());
//        
//        // Try to classify an instance using all models
//        Helper.classifyUsingModel(classifier_naivebayes, DATA_SOURCE_UNLABELED);
//        System.out.println("================================================");
//        Helper.classifyUsingModel(classifier_id3, DATA_SOURCE_UNLABELED);
//        System.out.println("================================================");
//        Helper.classifyUsingModel(classifier_j48, DATA_SOURCE_UNLABELED);
    }
}
