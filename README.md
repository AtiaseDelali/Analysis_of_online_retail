# Online Retail Analysis Project

## Project Overview
This project analyzes online retail data using SQL, PowerBI, and Python to derive business insights and create customer segmentation models. The analysis includes sales patterns, customer behavior, and predictive analytics.

## Technologies Used
- SQL (SQLite)
- Python
  - pandas
  - scikit-learn
  - lifetimes
- PowerBI

## Project Structure
```
online-retail-analysis/
├── data/
│   └── Online Retail.csv
├── sql/
│   └── analysis.sql
├── python/
│   └── customer_segmentation.py
├── powerbi/
│   └── retail_analysis.pbix
├── images/
│   ├── dashboard_screenshots/
│   └── analysis_outputs/
└── README.md
```

## Analysis Components

### 1. SQL Analysis
The SQL analysis includes:
- Sales performance metrics
- Geographic distribution of sales
- Product performance analysis
- Customer purchase patterns

### 2. PowerBI Dashboards
Three interconnected dashboards:
- Executive Overview
- Sales Performance Analysis
- Customer Insights

### 3. Python Customer Segmentation
Advanced customer segmentation using:
- RFM (Recency, Frequency, Monetary) Analysis
- K-means Clustering
- Customer Lifetime Value Prediction

## Key Insights

### Customer Segment Analysis
#### Summary Overview
Total Customers Analyzed: 4,338
#### Segment Distribution
-Active Loyal Low-Value: 1,788 customers (41.2%), $4,236.06 avg. spend
-Inactive Occasional High-Value: 2 customers (0.1%), $122,828.05 avg. spend
-Inactive Occasional Low-Value: 2,548 customers (58.7%), $428.44 avg. spend

#### Detailed Segment Analysis
##### Active Loyal Low-Value Segment Analysis
-Customer Base: 1,788 customers (41.2% of total)
Key Metrics:
-Recency: 34.8 days since last purchase
-Frequency: 8.0 purchases on average
-Monetary: $4,236.06 average total spend
-Average Order Value: $524.64
-Product Diversity: 111.5 unique products purchased
-Predicted Quarterly Value: $1,521.62
-Growth Potential Score: 7.8/100

Segment Characteristics: This segment represents your loyal customer base with recent purchases. These customers shop frequently and demonstrate strong engagement with your products. While loyal, these customers tend to make smaller purchases. Their value comes from consistency rather than transaction size.

##### Inactive Occasional High-Value Segment Analysis
Customer Base: 2 customers (0.1% of total)
Key Metrics:
-Recency: 163.5 days since last purchase
-Frequency: 1.5 purchases on average
-Monetary: $122,828.05 average total spend
-Average Order Value: $80,709.92
-Product Diversity: 2.0 unique products purchased
-Predicted Quarterly Value: $48,367.87
-Growth Potential Score: 11.0/100

Segment Characteristics: This segment represents customers who haven't purchased in a significant period. These customers made infrequent but large purchases and have since become inactive. They may be project-based buyers or seasonal shoppers who make large but infrequent purchases.

##### Inactive Occasional Low-Value Segment Analysis
Customer Base: 2,548 customers (58.7% of total)
Key Metrics:
-Recency: 133.0 days since last purchase
-Frequency: 1.7 purchases on average
-Monetary: $428.44 average total spend
-Average Order Value: $282.09
-Product Diversity: 26.4 unique products purchased
-Predicted Quarterly Value: $360.82
-Growth Potential Score: 1.4/100

Segment Characteristics: This segment represents customers who haven't purchased in a significant period. These customers made few, low-value purchases before becoming inactive. They likely had limited engagement with your brand from the beginning.

#### Strategic Recommendations
##### Strategic Recommendations for Active Loyal Low-Value Segment
Priority: MEDIUM - Frequency maintenance and value growth

Basket building incentives - Offer tiered discounts based on order size
Cross-sell campaigns - Suggest complementary products at checkout
Educational content - Share product usage ideas to increase utility
Limited-time promotions - Create urgency for additional purchases
Loyalty points acceleration - Offer bonus points for larger orders

##### Strategic Recommendations for Inactive Occasional High-Value Segment
Priority: MEDIUM - High-value recovery

VIP win-back program - Exclusive offers for returning customers
Personal outreach - Direct contact from customer service
Major product announcements - Share significant new offerings
Substantial incentives - Provide meaningful discounts to return
Account review - Ensure no service issues caused departure

##### Strategic Recommendations for Inactive Occasional Low-Value Segment
Priority: LOW - Selective reactivation

Final reactivation attempt - Last-chance special offer
New customer-like offers - Treat as essentially new to the brand
Low-cost engagement - Social media and email reconnection
Feedback survey - Learn about their departure for future improvement
Automated reactivation program - Periodic reminders with minimal resource investment

#### Key Performance Indicators
##### Key Performance Indicators for Active Loyal Low-Value
Universal KPIs:
-Segment size (customer count and percentage)
-Segment revenue contribution
-Segment profitability
-Segment-Specific KPIs:

Retention rate:
-Repeat purchase rate
-Average time between purchases
-Loyalty program participation rate
-Product category penetration
-Share of wallet (estimated)
-Average order value growth
-Cross-sell success rate

##### Key Performance Indicators for Inactive Occasional High-Value
Universal KPIs:
-Segment size (customer count and percentage)
-Segment revenue contribution
-Segment profitability

Segment-Specific KPIs:
-Reactivation rate
-Win-back campaign ROI
-Email deliverability
-Unsubscribe rate
-Recovery rate compared to acquisition cost
-Feedback survey completion rate

##### Key Performance Indicators for Inactive Occasional Low-Value
Universal KPIs:
-Segment size (customer count and percentage)
-Segment revenue contribution
-Segment profitability

Segment-Specific KPIs:
-Reactivation rate
-Win-back campaign ROI
-Email deliverability
-Unsubscribe rate

## Installation and Usage
1. Clone the repository
   ```bash
   git clone https://github.com/AtiaseDelali/Analysis_of_online_retail.git
   
   ```

2. Install required Python packages
   ```bash
   pip install -r requirements.txt
   ```

3. Run the SQL scripts using SQLite
   ```bash
   sqlite3 retail.db < sql/analysis.sql
   ```

4. Execute the Python analysis
   ```bash
   python python/customer_segmentation.py
   ```

## Results and Visualizations
[Add screenshots and descriptions of your visualizations here]

## Future Improvements
- Add time series forecasting
- Implement deep learning models
- Create interactive web dashboard

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
[Your Name] - [Your Email]
GitHub: [@yourusername](https://github.com/yourusername)

## Acknowledgments
- Dataset source: [Add source information]
- Special thanks to [Add any acknowledgments]
