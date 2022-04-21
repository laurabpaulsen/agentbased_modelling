# Agentbased_modelling

### Notes & ideas:
Investigating recruitment
* Vary what is taken into account
    * Does gender modulate the chance of getting promoted?
    * Is it modulated by who filled the position before?

* Kan det mere skiftende arbejdsmarked blive en fordel for kvinder?




Should the company be assumed to be growing, creating vacancies within each rank
"Vacancies at the entry level are filled through hiring from a hypothetical external pool of candidates, while vacancies at all other ranks are filled by promoting individuals from the rank immediately below, which, in turn, creates additional vacancies at the rank from which someone was promoted." - An Agent-Based Simulation of How Promotion Biases Impact Corporate Gender Diversity
* This is partially what is done now, except for the fact that a random agent is generated at the entry level
"For this version of the simulation we do not simulate direct hiring into higher ranks, nor do we allow promotions to skip levels. However, these and other details could be added if we were interested in exploring the effects of such modifications."
* We could potentially do this?

"By default, the promotion pool is set as a percentage of the total number of employees at that rank, and is based on the “promotion score” of each employee, a normalized value based on the amount of time the employee has spent at the current rank (“rank seniority”), relative to the amount of time spent by the most senior employee at that rank. In other words, in the absence of a bias, promotions are based on rank seniority, with some randomness to reflect the fact that seniority is not an exact indicator of merit."
* maybe clever enough to have a pool - we could test the implications of the shortlist article this way?


"When an employee is promoted, its seniority at the new rank is set to zero, while the simulation still tracks the total amount of time that the employee has been with the organization."
* This should potentially be done?

"Gender biases in the promotion process are simulated by adding a promotion-bias parameter, a value that is added to the rank seniority of each employee to calculate the promotion scores. The promotion-bias parameter can be positive or negative to simulate biases that favor men or women, respectively."





### Code to-do

    [] What kind of data do we need to save as we go?
    [X] Give weights in populate_company function rather than if statements in the function
    [] Should senority be 0 for randomly generated new agents? 
    [] How many employees
        "For all results reported here, the company begins with a total of 300 employees, 150 women and 150 men. The employees are distributed across the four levels in a way that roughly simulates a typical company: 40% at the entry level, 30% at the manager level, 20% at the VP level and 10% at the executive level"
    [] Annual churn rate








###

* forskellige tiltag hvor lang tid tager det?
* virksomhed, så mange parametre som muligt
    * kør med forskellige bias (+ til mændene, + til kvinderne, 0 dvs ingen forskel)
* flere kvinder på højere niveau, større sandsynlighed for at de søger
* med bias omkring at næste hire skal ligne den tidligere, og så fjerner man det

