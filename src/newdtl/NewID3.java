/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package newdtl;

import java.util.Enumeration;
import java.util.stream.DoubleStream;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;

public class NewID3 extends Classifier {

    private final double DOUBLE_MISSING_VALUE = Double.NaN;

    /**
     * The node's children.
     */
    private NewID3[] children;

    /**
     * Attribute used for splitting.
     */
    private Attribute splitAttribute;

    /**
     * Class value if node is leaf.
     */
    private double label;

    /**
     * Class distribution if node is leaf.
     */
    private double[] classDistributions;

    /**
     * Class attribute of dataset.
     */
    private Attribute classAttribute;

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Builds Id3 tree classifier.
     *
     * @param data the training data
     * @exception Exception if classifier failed to build
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {

        // Mengecek apakah data dapat dibuat classifier
        getCapabilities().testWithFail(data);

        // Menghapus instances dengan missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        makeTree(data);
    }

    /**
     * Creates an Id3 tree.
     *
     * @param data the training data
     * @exception Exception if tree failed to build
     */
    private void makeTree(Instances data) throws Exception {

        // Mengecek apakah tidak terdapat instance dalam node ini
        if (data.numInstances() == 0) {
            splitAttribute = null;
            label = DOUBLE_MISSING_VALUE;
            classDistributions = new double[data.numClasses()]; //???
        } else {
            // Mencari IG maksimum
            double[] infoGains = new double[data.numAttributes()];

            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                infoGains[att.index()] = computeInfoGain(data, att);
            }
            
            // cek max IG
            int maxIG = maxIndex(infoGains);
            if (maxIG != -1) {
                 splitAttribute = data.attribute(maxIndex(infoGains));
            } else {
                Exception exception = new Exception("array null");
                throw exception;
            }
           

            // Membuat daun jika IG-nya 0
            if (Double.compare(infoGains[splitAttribute.index()], 0) == 0) {
                splitAttribute = null;
                
                
                classDistributions = new double[data.numClasses()];
                for (int i = 0; i < data.numInstances(); i++) {
                    Instance inst = (Instance) data.instance(i);
                    classDistributions[(int) inst.classValue()]++;
                }

                normalizeClassDistribution();
                
                label = maxIndex(classDistributions);
                classAttribute = data.classAttribute();
            } else {
                // Membuat tree baru di bawah node ini
                Instances[] splitData = splitData(data, splitAttribute);
                children = new NewID3[splitAttribute.numValues()];
                for (int j = 0; j < splitAttribute.numValues(); j++) {
                    children[j] = new NewID3();
                    children[j].makeTree(splitData[j]);
                }
            }
        }
    }

    /**
     * Normalize the class distribution
     * @exception Exception if sum of class distribution is 0 or NAN
     */
    private void normalizeClassDistribution() throws Exception {
        double sum = DoubleStream.of(classDistributions).sum();

        if (!Double.isNaN(sum) && sum != 0) {
            for (int i = 0; i < classDistributions.length; ++i) {
                classDistributions[i] /= sum;
            }
        } else {
            Exception exception = new Exception("Class distribution: sum = 0 or NAN");
            throw exception;
        }
    }


    /**
     * Search for index with largest value from array of double
     *
     * @param array the array of double
     * @return index of array with maximum value, -1 if array empty
     */
    private static int maxIndex(double[] array) {
        int max = 0;
        
        if (array.length > 0) {
            for (int i = 1; i < array.length; ++i) {
                if (array[i] > array[max]) {
                    max = i;
                }
            }
            return max;
        } else {
            return -1;
        }
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    @Override
    public double classifyInstance(Instance instance)
        throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("NewID3: Cannot handle missing values");
        }
        if (splitAttribute == null) {
            return label;
        } else {
            return children[(int) instance.value(splitAttribute)].
                classifyInstance(instance);
        }
    }

    /**
     * Computes class distribution for instance using decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     * @throws NoSupportForMissingValuesException if instance has missing values
     */
    @Override
    public double[] distributionForInstance(Instance instance) // ga tau buat apa, ga dipanggil sama skali
        throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("NewID3: Cannot handle missing values");
        }
        if (splitAttribute == null) {
            return classDistributions;
        } else {
            return children[(int) instance.value(splitAttribute)].
                distributionForInstance(instance);
        }
    }

    /**
     * Prints the decision tree using the private toString method from below.
     *
     * @return a textual description of the classifier
     */
    @Override
    public String toString() {

        if ((classDistributions == null) && (children == null)) {
            return "NewID3: No model built yet.";
        }
        return "NewID3\n\n" + toString(0);
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     */
    private static double computeInfoGain(Instances data, Attribute att) {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double dataNumInstances = data.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                infoGain -= proportion * computeEntropy(splitdata);
            }
        }
        return infoGain;
    }

    /**
     * Computes the entropy of a dataset.
     *
     * @param data the data for which entropy is to be computed
     * @return the entropy of the data class distribution
     */
    private static double computeEntropy(Instances data) {

        double[] labelCounts = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); ++i) {
            labelCounts[(int) data.instance(i).classValue()]++;
        }

        double entropy = 0;
        for (int i = 0; i < labelCounts.length; i++) {
            if (labelCounts[i] > 0) {
                double proportion = labelCounts[i] / data.numInstances();
                entropy -= (proportion) * log2(proportion);
            }
        }
        return entropy;
    }

    /**
     * Count the logarithm value with base 2 of a number
     *
     * @param num number that will be counted
     * @return logarithm value with base 2
     */
    private static double log2(double num) {
        return (num == 0) ? 0 : Math.log(num) / Math.log(2);
    }

    /**
     * split the dataset based on nominal attribute
     *
     * @param data dataset used for splitting
     * @param att attribute used to split the dataset
     * @return array of instances which has been split by attribute
     */
    private static Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            splitData[(int) data.instance(i).value(att)].add(data.instance(i));
        }

        for (Instances splitData1 : splitData) {
            splitData1.compactify();
        }
        return splitData;
    }

    /**
     * Outputs a tree at a certain level.
     *
     * @param level the level at which the tree is to be printed
     * @return the tree as string at the given level
     */
    private String toString(int level) {

        StringBuilder text = new StringBuilder();

        if (splitAttribute == null) {
            if (Instance.isMissingValue(label)) {
                text.append(": null");
            } else {
                text.append(": ").append(classAttribute.value((int) label));
            }
        } else {
            for (int j = 0; j < splitAttribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(splitAttribute.name()).append(" = ").append(splitAttribute.value(j));
                text.append(children[j].toString(level + 1));
            }
        }
        return text.toString();
    }
}
