# Raw Dataset: Statlog German Credit Data

## Provenance
- **Source**: UCI Machine Learning Repository
- **Dataset**: Statlog (German Credit Data)
- **URL**: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
- **Instances**: 1,000 credit applications
- **Attributes**: 20 financial, personal, and loan attributes (7 numerical, 13 categorical)

## Target Variable
- Column: `credit_risk`
- Binary Classification:
  - `0`: Good Credit / Non-Default (700 applications, 70%)
  - `1`: Bad Credit / Default (300 applications, 30%)
*(Note: In the original UCI text file, 1 = Good and 2 = Bad. In `src.data.ingestion`, we map 1 -> 0 and 2 -> 1 to align with standard loss calculations where default is the positive minority class.)*

## Attribute Descriptions
1. **status_checking**: Status of existing checking account (`A11`: < 0 DM, `A12`: 0 to < 200 DM, `A13`: >= 200 DM / salary account, `A14`: no checking account).
2. **duration_months**: Duration of loan in months (numerical: 4 to 72).
3. **credit_history**: Credit history (`A30`: no credits/paid duly, `A31`: all credits at bank paid duly, `A32`: existing credits paid duly, `A33`: past delay, `A34`: critical account/other credits).
4. **purpose**: Loan purpose (`A40`: new car, `A41`: used car, `A42`: furniture/equipment, `A43`: radio/tv, `A44`: domestic appliances, `A45`: repairs, `A46`: education, `A48`: retraining, `A49`: business, `A410`: others).
5. **credit_amount**: Credit amount in Deutsche Mark (numerical: 250 to 18,424).
6. **savings_account**: Savings account/bonds (`A61`: < 100 DM, `A62`: 100 to < 500 DM, `A63`: 500 to < 1000 DM, `A64`: >= 1000 DM, `A65`: unknown/no savings).
7. **employment_since**: Present employment duration (`A71`: unemployed, `A72`: < 1 yr, `A73`: 1 to < 4 yrs, `A74`: 4 to < 7 yrs, `A75`: >= 7 yrs).
8. **installment_rate_pct**: Installment rate in percentage of disposable income (numerical: 1 to 4).
9. **personal_status_sex**: Personal status and sex (`A91`: male divorced/separated, `A92`: female divorced/separated/married, `A93`: male single, `A94`: male married/widowed, `A95`: female single).
10. **other_debtors**: Other debtors/guarantors (`A101`: none, `A102`: co-applicant, `A103`: guarantor).
11. **residence_since_years**: Present residence duration in years (numerical: 1 to 4).
12. **property**: Property assets (`A121`: real estate, `A122`: building society savings/life insurance, `A123`: car or other, `A124`: unknown / no property).
13. **age_years**: Age in years (numerical: 19 to 75).
14. **other_installment_plans**: Other installment plans (`A141`: bank, `A142`: stores, `A143`: none).
15. **housing**: Housing situation (`A151`: rent, `A152`: own, `A153`: for free).
16. **existing_credits**: Number of existing credits at this bank (numerical: 1 to 4).
17. **job**: Employment job category (`A171`: unemployed/unskilled non-resident, `A172`: unskilled resident, `A173`: skilled employee/official, `A174`: management/self-employed/highly qualified).
18. **people_liable**: Number of people liable for maintenance (numerical: 1 or 2).
19. **telephone**: Telephone registered (`A191`: none, `A192`: yes).
20. **foreign_worker**: Foreign worker status (`A201`: yes, `A202`: no).
