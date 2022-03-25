import pandas as pd
import numpy as np
import pulp


class CustomerSelection:
    """[summary]"""

    def __init__(self, dataframe, unit_cost, unit_profit, budget):

        self.dataframe = dataframe
        self.unit_cost = unit_cost
        self.unit_profit = unit_profit
        self.budget = budget
        self.problem = None
        self.decision_variables = []
        self.total_customers = ""
        self.total_cost = ""

    def definie_problem(self):  # Defining problem
        self.problem = pulp.LpProblem("SelectingBestCustomers", pulp.LpMaximize)

    def create_decision_variables(self):  # Creating decision variables
        for rownum, row in self.dataframe.iterrows():
            variablestr = str("x" + str(rownum))  # Create naming of variables
            variable = pulp.LpVariable(
                str(variablestr),
                lowBound=0,
                upBound=1,
                cat="Binary",  # Make variables binary
            )
            self.decision_variables.append(variable)

        print(
            "Total number of decision_variables: " + str(len(self.decision_variables))
        )

    def create_optimization_function(self):  # Create optimization function
        for rownum, row in self.dataframe.iterrows():
            for i, customer in enumerate(self.decision_variables):
                if rownum == i:
                    self.total_customers += (
                        self.unit_profit * row["acceptance_prob"] - self.unit_cost
                    ) * customer

        self.problem += self.total_customers

        print("Optimization function created")

    def create_constrain(
        self,
    ):  # Creating constrain - The budget must be grater then total contact cost
        for i, customer in enumerate(self.decision_variables):
            self.total_cost += self.unit_cost * customer

        self.problem += self.budget >= self.total_cost

        print("Constrain Created")

    def solve(self):  # running optimization
        optimization_result = self.problem.solve()
        assert optimization_result == pulp.LpStatusOptimal
        print("Status:", pulp.LpStatus[self.problem.status])
        print("Optimal Solution to the problem: ", pulp.value(self.problem.objective))

    def filter_dataframe(self):  # Storing decision variables result
        self.dataframe["selected_customer"] = 0

        for var in self.problem.variables():
            row_index = int(var.name[1:])
            self.dataframe.loc[row_index, "selected_customer"] = var.varValue

        self.dataframe = self.dataframe[["ID", "acceptance_prob", "selected_customer"]]
        self.dataframe = self.dataframe[self.dataframe["selected_customer"] == 1]

        print(
            "investing %i in the customers generated in the output dataframe, the estimated profit is %.2f"
            % (self.dataframe.shape[0] * 3, pulp.value(self.problem.objective))
        )

        return self.dataframe


def optimized_customer_selection(data, unit_cost, unit_profit, budget):

    optmizer = CustomerSelection(data, 3, 11, 6720)
    optmizer.definie_problem()
    optmizer.create_decision_variables()
    optmizer.create_optimization_function()
    optmizer.create_constrain()
    optmizer.solve()

    data = optmizer.filter_dataframe()
    expected_profit = pulp.value(optmizer.problem.objective)
    amount_invested = data.shape[0] * 3

    return data, expected_profit, amount_invested
