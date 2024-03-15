from functools import reduce
import numpy as np
from simpful import AutoTriangle
from .variable import Variable
from pydantic import BaseModel


class AutoRulesGenerator(BaseModel):
    
    inputs: list[Variable]
    output: list[Variable]
    values_io: list
    var_names:list
    # rules: list = None
    # weights: list = None
    
    @staticmethod
    def rules_database(
            self,
            inputs: list,
            output: list,
            values_io: list,
            var_names:list,
            # rules: list=None,
            # weights: list=None
        ) ->  tuple[list, list]:
        """
        This function build or append rules to the rules_database.

        Args:
            - inputs: The fuzzy input variables
            - output: The fuzzy output variables
            - values_io: List of values for each variable
            - rules: The set of rule
            - weights: The set of rules' weights.
        Returns:
            - rules
            - weights
        """
        # Inicialization of some variables
        rules = [] if rules is None else rules
        weights = [] if weights is None else weights
        terms_list = []
        weight_list = []

        # Join all vars inside an list
        fuzzy_vars = inputs.copy()
        fuzzy_vars.append(output[0])

        """ PASSO 2 """
        # Calculate the strogest pertinence considering the values_io
        for i in range(len(fuzzy_vars)):
            term, weight = self.strong_pertinence(fuzzy_vars[i], values_io[i])
            terms_list.append(term)
            weight_list.append(weight)

        """ PASSO 3 """
        # Rule creation and rules' weight calculation
        new_weight = reduce(lambda x, y: x * y, weight_list)
        new_rule = self.build_rule(inputs, output, terms_list,var_names)

        """ PASSO 4 """
        # If the rule is new, just add it to the rules' database
        if not any(new_rule[:new_rule.find('THEN')] in item for item in self.rules) or len(self.rules) == 0:
            self.rules.append(new_rule)
            weights.append(new_weight)
        # Else, if the rule is already in the database, add it if the new weight is higher than the old one
        else:
            arr = np.array(rules)
            mask = np.core.defchararray.find(arr.astype(str), new_rule[:new_rule.find('THEN')])
            old_weight = weights[mask[0]]
            if new_weight > old_weight:
                self.rules[mask[0]] = new_rule
                weights[mask[0]] = new_weight
        
        self.rules = rules
        self.weights = weights

    def strong_pertinence(
            self,
            var: Variable,
            value: float
        ) -> tuple[str, float]:
        """
        Determine the strongest pertinence region considering the value

        Args:
            - var: fuzzy variable
            - value: the test value
        Return
            - strong_term: the strongest term
            - value_term: the pertinence level
        """
        # Inicialization variables
        terms_list = []
        value_list = []

        # Calculation of highest pertinence
        for term, value in var.auto_triangle.get_values(value).items():
            terms_list.append(term)
            value_list.append(value)
        strong_term = terms_list[np.argmax(value_list)]
        value_term = value_list[np.argmax(value_list)]
        # strong_term = max(var.get_values(value), key= var.get_values(value).get)
        # value_term = var.get_values(value)[strong_term]
        return strong_term, value_term

    def build_rule(
            self,
            inputs:list,
            output:list,
            terms:list,
            var_names:list
        ) -> str:
        """
        Rule string creation to taylor with the simpful lib

        Args:
            - inputs: list of fuzzy variables of input
            - terms: list of the strongest terms for each fuzzy variable (inputs and outputs)
        Return:
            - rule_string: the rule string
        """
        rule_string =  "IF "
        for i in range(len(inputs)):
            rule_string += f"({var_names[i]} IS {terms[i]}) AND "
        rule_string = rule_string[:-4] + f"THEN ({var_names[-1]} IS {terms[-1]})"
        return rule_string

    def rules_compact(
            self,
            rules: list
        ):
        
        rule_output = set()
        for i in rules:
            rule_output.add(i[i.find("THEN"):])
        rule_output = list(rule_output)
        compact_rules = []

        for item_output in rule_output:
            string_compact = "IF ("
            indices = []
            for i, item in enumerate(rules):
                if item_output in item:
                    indices.append(i)

            for j in range(len(indices)):
                comb_ini = rules[indices[j]].find("IF (")+3
                comb_end = rules[indices[j]].find("THEN")-1
                string_compact+=rules[indices[j]][comb_ini:comb_end]
                string_compact+=") OR ("
                if j == len(indices)-1:
                    string_compact=string_compact[:-5]
                    string_compact+=" "+item_output
            compact_rules.append(string_compact)

        self.compact_rules = compact_rules