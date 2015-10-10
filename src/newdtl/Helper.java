package newdtl;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

public class Helper {

    /**
     * Constructor
     */
    public Helper() {
    }

    /**
     * Load the dataset from specified file into weka Instances
     *
     * @param file the file path to the dataset
     * @return
     */
    public static Instances loadDataFromFile(String file) {
        Instances data = null;

        try {
            data = DataSource.read(file);

            // setting class attribute if the data format does not provide this information
            // For example, the ARFF format saves the class attribute information as well
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return data;
    }

    /**
     * Remove the attributes from the dataset
     *
     * @param data
     * @param attribute a string representing the list of attributes. Since the
     * string will typically come from a user, attributes are indexed from 1.
     * eg: first-3,5,6-last
     * @return
     */
    public static Instances removeAttribute(Instances data, String attribute) {
        Instances newData = null;

        try {
            Remove remove = new Remove();
            remove.setAttributeIndices(attribute);
            remove.setInputFormat(data);
            newData = Filter.useFilter(data, remove);
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }

        return newData;
    }

    /**
     * Randomize the instances
     *
     * @param data the dataset that will be randomized
     * @return
     */
    public static Instances resample(Instances data) {
        Instances newData = null;

        try {
            Resample resample = new Resample();
            resample.setInputFormat(data);
            newData = Filter.useFilter(data, resample);
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }

        return newData;
    }

    /**
     * Build the classifier from dataset, allowed algorithms are naive bayes,
     * ID3, J48
     *
     * @param data the dataset that will be trained
     * @param type choice of algorithm, can be naivebayes, id3, or j48
     * @return
     */
    public static Classifier buildClassifier(Instances data, String type) {
        try {
            switch (type.toLowerCase()) {
                case "naivebayes":
                    NaiveBayes naiveBayes = new NaiveBayes();
                    naiveBayes.buildClassifier(data);

                    return naiveBayes;
                case "id3":
                    Id3 id3 = new Id3();
                    id3.buildClassifier(data);

                    return id3;
                case "newid3":
                    NewID3 newId3 = new NewID3();
                    newId3.buildClassifier(data);

                    return newId3;
                case "j48":
                    J48 j48 = new J48();
                    // j48.setOptions(new String[]{"-U"});
                    j48.buildClassifier(data);

                    return j48;
                case "newj48":
                    NewJ48 newJ48 = new NewJ48();
                    newJ48.buildClassifier(data);

                    return newJ48;
            }
        } catch (Exception ex) {
            Logger.getLogger(Helper.class.getName()).log(Level.SEVERE, null, ex);
        }

        return null;
    }

    /**
     * Do a ten fold cross validation using he model and instances
     *
     * @param data the dataset that will be used
     * @param classifier the classifier that will be used
     */
    public static void tenFoldCrossValidation(Instances data,
        Classifier classifier) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier, data,
                10, new Random(1));
            System.out
                .println(eval.toSummaryString("=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Function to test the classifier that has been built
     *
     * @param data the training set that will be used
     * @param classifier the classifier that will be used
     * @param datatest the test set that will be used
     */
    public static void testSetEvaluation(Instances data, Classifier classifier, Instances datatest) {
        try {
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(classifier, datatest);

            System.out
                .println(eval.toSummaryString("=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString());
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            // TODO Auto-generated catch block
        }
    }

    /**
     * Split instances into training data and test data to test the classifier
     *
     * @param data the dataset that will be split
     * @param classifier the classifier that will be used
     * @param percentage the percentage of the split, usually 80 which means 80%
     */
    public static void percentageSplit(Instances data, Classifier classifier, int percentage) {
        Instances dataSet = new Instances(data);
        dataSet.randomize(new Random(1));

        int trainSize = Math.round(dataSet.numInstances() * percentage / 100);
        int testSize = dataSet.numInstances() - trainSize;
        Instances trainSet = new Instances(dataSet, 0, trainSize);
        Instances testSet = new Instances(dataSet, trainSize, testSize);

        try {
            classifier.buildClassifier(trainSet);
            testSetEvaluation(trainSet, classifier, testSet);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Save a model that has been built to a file
     *
     * @param classifier the classifier that will be saved
     * @param file the filename for the file
     */
    public static void saveModelToFile(Classifier classifier, String file) {
        try {
            ObjectOutputStream oos = new ObjectOutputStream(
                new FileOutputStream(file));
            oos.writeObject(classifier);

            oos.flush();
            oos.close();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /**
     * Load a classifier model from file
     *
     * @param file the file path to the model file
     * @return
     */
    public static Classifier loadModelFromFile(String file) {
        Classifier cls = null;

        try {
            ObjectInputStream ois = new ObjectInputStream(new FileInputStream(
                file));
            cls = (Classifier) ois.readObject();

            ois.close();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

        return cls;
    }

    /**
     * Classify unlabeled instances in a file using a classifier
     *
     * @param classifier the classifier chosen to classify the instances
     * @param file the file path to the unlabeled instances
     */
    public static void classifyUsingModel(Classifier classifier, String file) {
        try {
            Instances unlabeled = DataSource.read(file);
            unlabeled.setClassIndex(unlabeled.numAttributes() - 1);

            Instances labeled = new Instances(unlabeled);

            // label instances
            for (int i = 0; i < unlabeled.numInstances(); i++) {
                double clsLabel = classifier.classifyInstance(unlabeled.instance(i));
                labeled.instance(i).setClassValue(clsLabel);
                System.out.println(labeled.instance(i));
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
}
