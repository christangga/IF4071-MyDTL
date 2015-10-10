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
import weka.core.Instance;
import weka.core.Instances;

public class NewJ48 extends Classifier {

    private final double DOUBLE_MISSING_VALUE = Double.NaN;
    private final double DOUBLE_ERROR_MAXIMUM = 1e-6;

    /**
     * The node's children.
     */
    private NewJ48[] children;

    /**
     * Attribute used for splitting.
     */
    private Attribute splitAttribute;

    /**
     * Threshold used for splitting if attribute is numeric.
     */
    private double splitThreshold;

    /**
     * Class value if node is leaf.
     */
    private double label;

    /**
     * Class distribution.
     */
    private double[] classDistributions;

    /**
     * Class attribute of dataset.
     */
    private Attribute classAttribute;

    /**
     * True if node is leaf.
     */
    private boolean isLeaf;

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
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    /**
     * Builds J48 tree classifier.
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
        pruneTree(data);
    }

    /**
     * Creates a J48 tree.
     *
     * @param data the training data
     * @exception Exception if tree failed to build
     */
    private void makeTree(Instances data) throws Exception {

        // Mengecek apakah tidak terdapat instance dalam node ini
        if (data.numInstances() == 0) {
            splitAttribute = null;
            label = DOUBLE_MISSING_VALUE;
            classDistributions = new double[data.numClasses()];
            isLeaf = true;
        } else {
            // Mencari Gain Ratio maksimum
            double[] gainRatios = new double[data.numAttributes()];
            double[] thresholds = new double[data.numAttributes()];

            Enumeration attEnum = data.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                double[] result = computeGainRatio(data, att);
                gainRatios[att.index()] = result[0];
                thresholds[att.index()] = result[1];
            }

            splitAttribute = data.attribute(maxIndex(gainRatios));
            splitThreshold = thresholds[maxIndex(gainRatios)];

            classDistributions = new double[data.numClasses()];
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = (Instance) data.instance(i);
                classDistributions[(int) inst.classValue()]++;
            }

            // Membuat daun jika Gain Ratio-nya 0
            if (doubleEqual(gainRatios[splitAttribute.index()], 0)) {
                splitAttribute = null;

                // normalizeDouble(classDistributions);
                label = maxIndex(classDistributions);
                classAttribute = data.classAttribute();
                isLeaf = true;
            } else {
                // Mengecek jika ada missing value
                if (isMissing(data, splitAttribute)) {
                    // cari modus
                    int index = modusIndex(data, splitAttribute);

                    // ubah data yang punya missing value
                    Enumeration dataEnum = data.enumerateInstances();
                    while (dataEnum.hasMoreElements()) {
                        Instance inst = (Instance) dataEnum.nextElement();
                        if (inst.isMissing(splitAttribute)) {
                            inst.setValue(splitAttribute, splitAttribute.value(index));
                        }
                    }
                }

                // Membuat tree baru di bawah node ini
                Instances[] splitData;
                if (splitThreshold > 0) {
                    splitData = splitData(data, splitAttribute, splitThreshold);
                    children = new NewJ48[2];
                    for (int j = 0; j < 2; j++) {
                        children[j] = new NewJ48();
                        children[j].makeTree(splitData[j]);
                    }
                } else {
                    splitData = splitData(data, splitAttribute);
                    children = new NewJ48[splitAttribute.numValues()];
                    for (int j = 0; j < splitAttribute.numValues(); j++) {
                        children[j] = new NewJ48();
                        children[j].makeTree(splitData[j]);
                    }
                }
                isLeaf = false;
            }
        }
    }

    /**
     * Creates a pruned J48 tree using expected error pruning.
     *
     * @param data the training data
     */
    private double pruneTree(Instances data) throws Exception {

        double staticError = staticErrorEstimate((int) DoubleStream.of(classDistributions).sum(),
            (int) classDistributions[maxIndex(classDistributions)], classDistributions.length);

        if (isLeaf) {
            return staticError;
        } else {
            double backupError = 0;
            double totalInstances = DoubleStream.of(classDistributions).sum();

            for (NewJ48 children1 : children) {
                double totalChildInstances = DoubleStream.of(children1.classDistributions).sum();
                backupError += totalChildInstances / totalInstances * children1.pruneTree(data);
            }

            if (staticError < backupError) {
                splitAttribute = null;
                label = maxIndex(classDistributions);
                classAttribute = data.classAttribute();
                isLeaf = true;
                children = null;

                return staticError;
            } else {
                return backupError;
            }
        }
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {

        if (splitAttribute == null) {
            return label;
        } else {
            if (splitThreshold > 0) {
                if (instance.value(splitAttribute) <= splitThreshold) {
                    return children[0].classifyInstance(instance);
                } else {
                    return children[1].classifyInstance(instance);
                }
            } else {
                return children[(int) instance.value(splitAttribute)].
                    classifyInstance(instance);
            }
        }
    }

    /**
     * Computes class distribution for instance using decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {

        if (splitAttribute == null) {
            return classDistributions;
        } else {
            if (splitThreshold > 0) {
                if (instance.value(splitAttribute) <= splitThreshold) {
                    return children[0].distributionForInstance(instance);
                } else {
                    return children[1].distributionForInstance(instance);
                }
            } else {
                return children[(int) instance.value(splitAttribute)].
                    distributionForInstance(instance);
            }
        }
    }

    /**
     * split the dataset based on attribute
     *
     * @param data dataset used for splitting
     * @param att attribute used to split the dataset
     * @return
     */
    private Instances[] splitData(Instances data, Attribute att) {

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
     * split the dataset based on attribute for numeric attribute
     *
     * @param data dataset used for splitting
     * @param att attribute used to split the dataset
     * @param threshold the threshold value
     * @return
     */
    private Instances[] splitData(Instances data, Attribute att, double threshold) {

        Instances[] splitData = new Instances[2];
        for (int j = 0; j < 2; j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }

        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).value(att) <= threshold) {
                splitData[0].add(data.instance(i));
            } else {
                splitData[1].add(data.instance(i));
            }
        }

        for (Instances splitData1 : splitData) {
            splitData1.compactify();
        }

        return splitData;
    }

    /**
     * Computes Gain Ratio for an attribute.
     *
     * @param data the data for which gain ratio is to be computed
     * @param att the attribute
     * @return the gain ratio for the given attribute and data
     * @throws Exception if computation fails
     */
    private double[] computeGainRatio(Instances data, Attribute att)
        throws Exception {

        if (att.isNumeric()) {
            data.sort(att);
            double[] threshold = new double[data.numInstances() - 1];
            double[] gainRatios = new double[data.numInstances() - 1];
            for (int i = 0; i < data.numInstances() - 1; i++) {
                threshold[i] = data.instance(i).value(att);
                double infoGain = computeInfoGain(data, att, threshold[i]);
                double splitInfo = computeSplitInformation(data, att, threshold[i]);
                gainRatios[i] = infoGain > 0 ? infoGain / splitInfo : infoGain;
            }
            if (threshold.length > 0) {
                return new double[]{gainRatios[maxIndex(gainRatios)], threshold[maxIndex(gainRatios)]};
            } else {
                return new double[]{0, 0};
            }
        } else {
            double infoGain = computeInfoGain(data, att);
            double splitInfo = computeSplitInformation(data, att);

            return new double[]{splitInfo > 0 ? infoGain / splitInfo : splitInfo, 0};
        }
    }

    /**
     * Computes information gain for an attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     * @throws Exception if computation fails
     */
    private double computeInfoGain(Instances data, Attribute att)
        throws Exception {

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
     * Computes information gain for a numeric attribute.
     *
     * @param data the data for which info gain is to be computed
     * @param att the attribute
     * @return the information gain for the given attribute and data
     * @throws Exception if computation fails
     */
    private double computeInfoGain(Instances data, Attribute att, double threshold)
        throws Exception {

        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att, threshold);
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
     * @throws Exception if computation fails
     */
    private double computeEntropy(Instances data) throws Exception {

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
     * Computes Split information for an attribute.
     *
     * @param data the data for which split information is to be computed
     * @param att the attribute
     * @return the split information for the given attribute and data
     * @throws Exception if computation fails
     */
    private double computeSplitInformation(Instances data, Attribute att) throws Exception {

        double splitInfo = 0;
        Instances[] splitData = splitData(data, att);
        double dataNumInstances = data.numInstances();

        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                splitInfo -= proportion * log2(proportion);
            }
        }
        return splitInfo;
    }

    /**
     * Computes Split information for a numeric attribute.
     *
     * @param data the data for which split information is to be computed
     * @param att the attribute
     * @return the split information for the given attribute and data
     * @throws Exception if computation fails
     */
    private double computeSplitInformation(Instances data, Attribute att, double threshold) throws Exception {

        double splitInfo = 0;
        Instances[] splitData = splitData(data, att, threshold);
        double dataNumInstances = data.numInstances();

        for (Instances splitdata : splitData) {
            if (splitdata.numInstances() > 0) {
                double splitNumInstances = splitdata.numInstances();
                double proportion = splitNumInstances / dataNumInstances;
                splitInfo -= proportion * log2(proportion);
            }
        }
        return splitInfo;
    }

    public double staticErrorEstimate(int N, int n, int k) {

        double E = (N - n + k - 1) / (double) (N + k);

        return E;
    }

    public double backUpError() {

        double E = 0;
        double totalInstances = DoubleStream.of(classDistributions).sum();
        for (NewJ48 child : children) {
            double totalChildInstances = DoubleStream.of(child.classDistributions).sum();
            E += totalChildInstances / totalInstances
                * staticErrorEstimate((int) totalChildInstances, (int) child.classDistributions[(int) child.label], child.classDistributions.length);

        }

        return E;
    }

    /**
     * search data that has missing value for attribute
     *
     * @param data the data for searching
     * @param att the attribute for searching
     * @return if data has missing value for attribute
     */
    private boolean isMissing(Instances data, Attribute att) {

        boolean isMissingValue = false;
        Enumeration dataEnum = data.enumerateInstances();

        while (dataEnum.hasMoreElements() && !isMissingValue) {
            Instance inst = (Instance) dataEnum.nextElement();
            if (inst.isMissing(att)) {
                isMissingValue = true;
            }
        }

        return isMissingValue;
    }

    /**
     * search index of attribute that has most common value
     *
     * @param data the data for searching
     * @param att the attribute for searching
     * @return index of attribute that has most common value
     */
    private int modusIndex(Instances data, Attribute att) {

        // cari modus
        int[] modus = new int[att.numValues()];
        Enumeration dataEnumeration = data.enumerateInstances();

        while (dataEnumeration.hasMoreElements()) {
            Instance inst = (Instance) dataEnumeration.nextElement();
            if (!inst.isMissing(att)) {
                modus[(int) inst.value(att)]++;
            }
        }

        // cari modus terbesar
        int max = 0;
        int index = -1;
        for (int i = 0; i < modus.length; ++i) {
            if (modus[i] > max) {
                max = modus[i];
                index = i;
            }
        }

        return index;
    }

    /**
     * Prints the decision tree using the private toString method from below.
     *
     * @return a textual description of the classifier
     */
    @Override
    public String toString() {

        if ((classDistributions == null) && (children == null)) {
            return "NewJ48: No model built yet.";
        }
        return "NewJ48 pruned tree\n------------------\n" + toString(0);
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

                double totalInstances = 0;
                for (int i = 0; i < classDistributions.length; i++) {
                    totalInstances += classDistributions[i];
                }
                text.append(" (").append(totalInstances);

                double wrongClass = totalInstances - classDistributions[(int) label];
                if (wrongClass > 0) {
                    text.append("/").append(totalInstances - classDistributions[(int) label]);
                }

                text.append(")");
            }
        } else {
            if (splitAttribute.isNumeric()) {
                for (int j = 0; j < 2; j++) {
                    text.append("\n");
                    for (int i = 0; i < level; i++) {
                        text.append("|  ");
                    }
                    if (j == 0) {
                        text.append(splitAttribute.name()).append(" <= ").append(splitThreshold);
                    } else {
                        text.append(splitAttribute.name()).append(" > ").append(splitThreshold);
                    }
                    text.append(children[j].toString(level + 1));
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
        }
        
        return text.toString();
    }

    /**
     * Check whether two double values are the same
     *
     * @param d1 the first double value
     * @param d2 the second double value
     * @return true if the values are the same, false if not
     */
    private boolean doubleEqual(double d1, double d2) {

        return (d1 == d2) || Math.abs(d1 - d2) < DOUBLE_ERROR_MAXIMUM;
    }

    /**
     * Count the logarithm value with base 2 of a number
     *
     * @param num number that will be counted
     * @return logarithm value with base 2
     */
    private double log2(double num) {

        return (num == 0) ? 0 : Math.log(num) / Math.log(2);
    }

    /**
     * Search for index with largest value from array of double
     *
     * @param array the array of double
     * @return index of array with maximum value
     */
    private int maxIndex(double[] array) {

        double max = 0;
        int index = 0;

        if (array.length > 0) {
            for (int i = 0; i < array.length; ++i) {
                if (array[i] > max) {
                    max = array[i];
                    index = i;
                }
            }
            return index;
        } else {
            return -1;
        }
    }

    /**
     * Normalize the values in array of double
     *
     * @param array the array of double
     */
    private void normalizeDouble(double[] array) {

        double sum = 0;
        for (double d : array) {
            sum += d;
        }

        if (!Double.isNaN(sum) && sum != 0) {
            for (int i = 0; i < array.length; ++i) {
                array[i] /= sum;
            }
        } else {
            // Do nothing
        }
    }

}
