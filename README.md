# AERO4450
Project associated with AERO4450 (UQ Aerospace Engineering Course). This project focuses on the combustion modelling of a scramjet engine (see attached pdf file for task sheet). Numerical methods were implemented to model the unsteady flamelet model, which was used to analyse the mass fraction of the combustion product vs. the mixture fraction.
Firstly, an implicit scheme was derived for this flamelet model and it could be used to find the steady-state distribution of the combustion product mass fraction for a given value of the scalar dissipation constant (N). 

The golden section search method was then implemented to find the critical value of this scalar dissipation rate, where values above this force the combustion products to become extinct.
