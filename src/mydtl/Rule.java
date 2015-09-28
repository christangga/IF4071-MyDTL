package mydtl;

import java.util.List;

public class Rule {
    private List<AttValue> attValue;
    private double label;

    public List<AttValue> getAttValue() {
        return attValue;
    }

    public void setAttValue(List<AttValue> attValue) {
        this.attValue = attValue;
    }

    public double getLabel() {
        return label;
    }

    public void setLabel(double label) {
        this.label = label;
    }
    
    private class AttValue {
        private String attribute;
        private String value;

        public String getAttribute() {
            return attribute;
        }

        public void setAttribute(String attribute) {
            this.attribute = attribute;
        }

        public String getValue() {
            return value;
        }

        public void setValue(String value) {
            this.value = value;
        }
    }
}
