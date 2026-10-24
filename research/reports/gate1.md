# Gate 1 report

Data: 2015-01-02 to 2026-10-23, 3324 tickers (delisted included).

## Share-basis events picked up

1677 days where the raw close jumps but KRX's adjusted change does not (splits, mergers, rights).

## Universe vs indices (annualised, gross)

| series | CAGR | vol | MDD |
|---|---|---|---|
| EW universe | -0.2% | 23.6% | -44.6% |
| KOSPI | 12.7% | 24.6% | -33.6% |
| KOSDAQ | 1.8% | 24.8% | -41.4% |

Monthly correlation EW universe vs KOSPI: 0.69

## Cap-weighted rebuild vs official index (monthly, open to open)

| market | correlation | CAGR ours | CAGR index | tracking error |
|---|---|---|---|---|
| KOSPI | 0.9985 | 13.2% | 12.7% | 1.4% |
| KOSDAQ | 0.9969 | 1.1% | 1.8% | 2.0% |

## Factor results (top 20% equal weight, monthly, cost 0.25% one-way, from 2016-01-01)

| signal | CAGR | excess vs EW | IR | MDD | turnover/mo | Q5-Q1 ann | t |
|---|---|---|---|---|---|---|---|
| momentum_12_1 | -2.8% | -1.3% | -0.12 | -53.7% | 66.9% | 6.5% | 1.23 |
| reversal_1m | -5.9% | -4.9% | -0.58 | -66.7% | 167.2% | 8.4% | 1.65 |
| low_vol | 3.5% | 2.3% | 0.21 | -37.7% | 54.9% | 21.4% | 3.49 |
| small_size | -6.4% | -5.7% | -0.57 | -76.8% | 107.2% | -8.8% | -1.60 |

## Quintile returns (annualised, gross; Q1 = lowest score)

| signal | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| momentum_12_1 | -3.0% | 2.4% | 3.4% | 6.9% | 3.5% |
| reversal_1m | -5.5% | 5.5% | 6.3% | 4.3% | 2.8% |
| low_vol | -14.9% | 2.8% | 9.5% | 9.3% | 6.6% |
| small_size | 8.9% | 6.2% | 2.5% | -4.4% | 0.1% |
