import wrds
conn = wrds.Connection()
conn.list_libraries().sort()
type(conn.list_libraries())
conn.list_tables(library='comp')

list_sorted = sorted(conn.list_libraries())
print(list_sorted)
# Extract first 5 obs from comp.company
list_table = conn.list_tables(library="compg")
print(list_table)

company = conn.get_table(library='comp', table='company', obs=5)
company.shape
# Narrow down the specific columns to extract

company_narrow = conn.get_table(library='comp', table='company', 
                                columns = ['conm', 'gvkey', 'cik'], obs=5)
company_narrow.shape
company_narrow
# Select one stock's monthly price
# from 2019 onwards

apple = conn.raw_sql("""select permno, date, prc, ret, shrout 
                        from crsp.msf 
                        where permno = 14593
                        and date>='01/01/2019'""", 
                     date_cols=['date'])

apple 
apple.dtypes
apple_fund = conn.raw_sql("""select a.gvkey, a.iid, a.datadate, a.tic, a.conm,
                            a.at, b.prccm, b.cshoq 
                            
                            from comp.funda a 
                            inner join comp.secm b 
                            
                            on a.gvkey = b.gvkey
                            and a.iid = b.iid
                            and a.datadate = b.datadate
                        
                            where a.tic = 'AAPL' 
                            and a.datadate>='01/01/2010'
                            and a.datafmt = 'STD' 
                            and a.consol = 'C' 
                            and a.indfmt = 'INDL'
                            """, date_cols=['datadate'])

apple_fund.shape
apple_fund
import pandas as pd

# export the dataframe to csv format
apple_fund.to_excel('apple_fund.xlsx')