--import tables for Covidproject from Excel
--quick view of data from tables 
SELECT *
FROM coviddeaths

SELECT *
FROM covidvaccinations
ORDER BY 2, 3

--the output is ordered in ascending form according to location and date
SELECT location, date, total_cases, new_cases, total_deaths, population
FROM coviddeaths
ORDER BY 1, 2

--Maximum, minimum and average number of deaths
SELECT continent, MAX(total_deaths) AS Maximumdeaths, MIN(total_deaths) AS Minimumdeaths, AVG(total_deaths) AS AverageDeaths
FROM coviddeaths
WHERE continent IS NOT NULL
GROUP BY continent

--Total cases vs total deaths
--Your likelihood of dying if you contract covid
SELECT location, date, total_cases, total_deaths, (total_deaths / total_cases)*100 AS DeathsPercentage 
FROM coviddeaths
WHERE continent IS NOT NULL
ORDER BY 1, 2

--In Kenya
SELECT location, date, total_cases, total_deaths, (total_deaths / total_cases)*100 AS DeathsPercentage 
FROM coviddeaths
WHERE location like '%Kenya%'
ORDER BY 1, 2

--In USA
SELECT location, date, total_cases, total_deaths, (total_deaths / total_cases)*100 AS DeathsPercentage 
FROM coviddeaths
WHERE location like '%States%'
ORDER BY 1, 2

--In Yemen
SELECT location, date, total_cases, total_deaths, (total_deaths / total_cases)*100 AS DeathsPercentage 
FROM coviddeaths
WHERE location like '%Yemen%'
ORDER BY 1, 2

--In Asia
SELECT continent, location, date, total_cases, total_deaths, (total_deaths / total_cases)*100 AS DeathsPercentage 
FROM coviddeaths
WHERE continent like '%Asia%'
ORDER BY 1, 2

--total cases vs population
--what % of population cotracted the virus
SELECT location, date, total_cases, population, (total_cases / population)*100 AS CasePercentage 
FROM coviddeaths
WHERE continent IS NOT NULL
ORDER BY 1, 2

--In Kenya
SELECT location, date, population, total_cases, (total_cases / population)*100 AS CasePercentage 
FROM coviddeaths
WHERE location like '%Kenya%'
ORDER BY 1, 2

--In Africa
SELECT continent,location, date, total_cases, (total_cases / population)*100 AS CasePercentage
FROM coviddeaths
WHERE continent like '%Africa%'
ORDER BY 1, 2

--continents with highest infection rate compared to population
SELECT continent, population, MAX(total_cases) AS HighestInfections, MAX((total_cases / population)) *100 AS InfectedPercentage
FROM coviddeaths
WHERE continent IS NOT NULL
GROUP BY continent, population 
ORDER BY InfectedPercentage DESC

--countries in Africa with highest infection rate compared to population in Africa
SELECT location, population, MAX(total_cases) AS HighestInfections, MAX((total_cases / population)) *100 AS InfectedPercentage
FROM coviddeaths
WHERE continent like '%Africa%'
GROUP BY location, population 
ORDER BY InfectedPercentage DESC

--countries with highest death count as compared to population in Africa
SELECT location, population, MAX(total_deaths) AS TotalDeathCount
FROM coviddeaths
WHERE continent like '%Africa%'
GROUP BY location, population 
ORDER BY TotalDeathCount DESC

--countries with highest death count as compared to population
SELECT location, population, MAX(total_deaths) AS TotalDeathCount
FROM coviddeaths
WHERE continent IS NOT NULL
GROUP BY location, population
ORDER BY TotalDeathCount DESC

--continents with highest death count as compared to population
SELECT continent, MAX(total_deaths) AS TotalDeathCount
FROM coviddeaths
WHERE continent IS NOT NULL
GROUP BY continent
ORDER BY TotalDeathCount DESC

--Global numbers by date
SELECT date, SUM(new_cases)AS SumofNewCases, SUM(new_deaths) AS SumofNewDeaths
FROM coviddeaths
WHERE continent IS NOT NULL
GROUP BY date
ORDER BY 2 DESC

SELECT SUM(new_cases)AS TotalCases, SUM(new_deaths) AS TotalDeaths
FROM coviddeaths
WHERE continent IS NOT NULL
ORDER BY 2 DESC



--Joining the coviddeaths table and the covidvaccinations table (using commom location & date)
SELECT *
FROM coviddeaths
JOIN covidvaccinations
ON coviddeaths.location = covidvaccinations.location
AND coviddeaths.date = covidvaccinations.date


--Total population vs vaccinations
--how many people got vaccinated
--dt is short for table coviddeaths and vc is short for table covidvaccinations
SELECT dt.continent, dt.location, dt.date, dt.population, vc.new_vaccinations
FROM coviddeaths dt
JOIN covidvaccinations vc
ON dt.location = vc.location
AND dt.date = vc.date
WHERE dt.continent IS NOT NULL
ORDER BY 1,2,3

 --adding new vaccinations consequtively by location
SELECT dt.continent, dt.location, dt.date, dt.population, vc.new_vaccinations, SUM(vc.new_vaccinations) OVER (Partition by dt.location ORDER BY dt.location, dt.date) AS RollingNumberofVaccinations
FROM coviddeaths dt
JOIN covidvaccinations vc
ON dt.location = vc.location
AND dt.date = vc.date
WHERE dt.continent IS NOT NULL
ORDER BY 1,2,3


--Using CTE
WITH popvsvac(continent, location, date, population, new_vaccinations, RollingNumberofVaccinations)AS
(
SELECT dt.continent, dt.location, dt.date, dt.population, vc.new_vaccinations, SUM(vc.new_vaccinations) OVER (Partition by dt.location ORDER BY dt.location, dt.date) AS RollingNumberofVaccinations
FROM coviddeaths dt
JOIN covidvaccinations vc
ON dt.location = vc.location
AND dt.date = vc.date
WHERE dt.continent IS NOT NULL
)
SELECT *, (RollingNumberofVaccinations/population) *100
FROM popvsvac


--Using Temp Table

DROP TABLE IF EXISTS #PercentPopVaccinated

CREATE TABLE #PercentPopVaccinated
(
continent nvarchar(255),
location nvarchar(255),
Date datetime,
Population float,
new_vaccinations float,
RollingNumberofVaccinations float
)
INSERT INTO #PercentPopVaccinated
SELECT dt.continent, dt.location, dt.date, dt.population, vc.new_vaccinations, SUM(vc.new_vaccinations) OVER (Partition by dt.location ORDER BY dt.location, dt.date) AS RollingNumberofVaccinations
FROM coviddeaths dt
JOIN covidvaccinations vc
ON dt.location = vc.location
AND dt.date = vc.date
WHERE dt.continent IS NOT NULL

SELECT *, (RollingNumberofVaccinations/population) *100
FROM #PercentPopVaccinated



--Creating views for later vizualiations

CREATE VIEW PercentPopVaccinated AS
SELECT dt.continent, dt.location, dt.date, dt.population, vc.new_vaccinations, SUM(vc.new_vaccinations) OVER (Partition by dt.location ORDER BY dt.location, dt.date) AS RollingNumberofVaccinations
FROM coviddeaths dt
JOIN covidvaccinations vc
ON dt.location = vc.location
AND dt.date = vc.date
WHERE dt.continent IS NOT NULL

CREATE VIEW Global AS
SELECT SUM(new_cases)AS TotalCases, SUM(new_deaths) AS TotalDeaths
FROM coviddeaths
WHERE continent IS NOT NULL

CREATE VIEW AfricaInfections AS
SELECT location, population, MAX(total_cases) AS HighestInfections, MAX((total_cases / population)) *100 AS InfectedPercentage
FROM coviddeaths
WHERE continent like '%Africa%'
GROUP BY location, population 

CREATE VIEW Death AS
SELECT location, date, total_cases, total_deaths, (total_deaths / total_cases)*100 AS DeathsPercentage 
FROM coviddeaths
WHERE continent IS NOT NULL
