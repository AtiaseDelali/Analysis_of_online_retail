-- Basic sales analysis
-- Total revenue by country
SELECT 
    Country,
    ROUND(SUM(Quantity * UnitPrice), 2) as TotalRevenue,
    COUNT(DISTINCT CustomerID) as NumberOfCustomers,
    COUNT(DISTINCT InvoiceNo) as NumberOfOrders
FROM online_retail
WHERE Quantity > 0
GROUP BY Country
ORDER BY TotalRevenue DESC;

-- Monthly revenue trend
SELECT 
    strftime('%Y-%m', InvoiceDate) as Month,
    ROUND(SUM(Quantity * UnitPrice), 2) as MonthlyRevenue,
    COUNT(DISTINCT InvoiceNo) as NumberOfOrders,
    COUNT(DISTINCT CustomerID) as NumberOfCustomers
FROM online_retail
WHERE Quantity > 0
GROUP BY Month
ORDER BY Month;

-- Top selling products
SELECT 
    StockCode,
    Description,
    SUM(Quantity) as TotalQuantitySold,
    ROUND(SUM(Quantity * UnitPrice), 2) as TotalRevenue,
    COUNT(DISTINCT CustomerID) as NumberOfUniqueCustomers
FROM online_retail
WHERE Quantity > 0
GROUP BY StockCode, Description
ORDER BY TotalQuantitySold DESC
LIMIT 10;

-- Customer purchase frequency
SELECT 
    CustomerID,
    COUNT(DISTINCT InvoiceNo) as NumberOfOrders,
    ROUND(SUM(Quantity * UnitPrice), 2) as TotalSpent,
    ROUND(AVG(Quantity * UnitPrice), 2) as AverageOrderValue,
    COUNT(DISTINCT  substr(InvoiceDate, 6, 2)) as NumberOfActiveMonths
FROM online_retail
WHERE Quantity > 0 AND CustomerID IS NOT NULL
GROUP BY CustomerID
ORDER BY TotalSpent DESC;

-- Product category analysis (based on first word of description)
WITH ProductCategories AS (
    SELECT 
        SUBSTR(Description, 1, INSTR(Description || ' ', ' ')-1) as Category,
        COUNT(*) as ProductCount,
        ROUND(SUM(Quantity * UnitPrice), 2) as CategoryRevenue
    FROM online_retail
    WHERE Quantity > 0
    GROUP BY SUBSTR(Description, 1, INSTR(Description || ' ', ' ')-1)
)
SELECT *
FROM ProductCategories
ORDER BY CategoryRevenue DESC
LIMIT 15;
