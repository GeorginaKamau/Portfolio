SELECT *
FROM Nashville

--taking out the time in date
SELECT SaleDateC, CONVERT(Date, SaleDate)
FROM Nashville

ALTER TABLE Nashville
ADD SaleDateC Date;

--Delete previous SaleDate column
ALTER TABLE Nashville
DROP COLUMN SaleDate


UPDATE Nashville
SET SaleDateC = CONVERT(Date, SaleDate)

--replacing null values in property adress with addresses with same parcelID
--isnull checks to see if the first column entry is null then replaces it with the second column entry
SELECT a.ParcelID, a.PropertyAddress, a.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress, b.PropertyAddress)
--replicating table and giving it different aliases
FROM [Housing data].dbo.Nashville a
JOIN [Housing data].dbo.Nashville b
ON a.ParcelID = b.ParcelID
AND a.[UniqueID ] <> b.[UniqueID ]
WHERE a.PropertyAddress IS NULL

--update original table a
UPDATE a
SET propertyAddress = ISNULL(a.PropertyAddress, b.PropertyAddress)
FROM [Housing data].dbo.Nashville a
JOIN [Housing data].dbo.Nashville b
ON a.ParcelID = b.ParcelID
AND a.[UniqueID ] <> b.[UniqueID ]
WHERE a.PropertyAddress IS NULL

--breaking the address into individual columns (address, city, state)
SELECT PropertyAddress
FROM Nashville

SELECT 
SUBSTRING(PropertyAddress, 1, CHARINDEX (',' , PropertyAddress) -1) AS Address,
SUBSTRING(PropertyAddress, CHARINDEX (',' , PropertyAddress) +1, LEN(PropertyAddress)) AS Address
FROM Nashville

--Creating 2 new columns
ALTER TABLE Nashville
ADD SplitAddress NVARCHAR(255);


UPDATE Nashville
SET SplitAddress = SUBSTRING(PropertyAddress, 1, CHARINDEX (',' , PropertyAddress) -1)

ALTER TABLE Nashville
ADD SplitCity NVARCHAR(255);

UPDATE Nashville
SET SplitCity = SUBSTRING(PropertyAddress, CHARINDEX (',' , PropertyAddress) +1, LEN(PropertyAddress))

SELECT OwnerAddress
FROM Nashville

SELECT 
PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3),
PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2),
PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1)
FROM Nashville

ALTER TABLE Nashville
ADD OwnerSplitAddress NVARCHAR(255);


UPDATE Nashville
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 3)

ALTER TABLE Nashville
ADD OwnerSplitCity NVARCHAR(255);

UPDATE Nashville
SET OwnerSplitCity = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 2)

ALTER TABLE Nashville
ADD OwnerSplitState NVARCHAR(255);

UPDATE Nashville
SET OwnerSplitState = PARSENAME(REPLACE(OwnerAddress, ',', '.'), 1)


SELECT DISTINCT(SoldAsVacant), COUNT(SoldAsVacant)
FROM Nashville
GROUP BY SoldAsVacant
ORDER BY 2
--convert Y and N to YES and NO respectively

SELECT SoldAsVacant
, CASE WHEN SoldAsVacant = 'Y' THEN 'Yes'
	   WHEN SoldAsVacant = 'N' THEN 'No'
	   ELSE SoldAsVacant
	   END
FROM Nashville

UPDATE Nashville
SET SoldAsVacant = CASE WHEN SoldAsVacant = 'Y' THEN 'Yes'
	   WHEN SoldAsVacant = 'N' THEN 'No'
	   ELSE SoldAsVacant
	   END


--remove duplicate entries
SELECT *,
	ROW_NUMBER()OVER(
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDateC,
				 LegalReference
				 ORDER BY 
				 UniqueID) Row_run
FROM Nashville
ORDER BY ParcelID

--seeing the duplicates
WITH RowNumCTE AS(
SELECT *,
	ROW_NUMBER()OVER(
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDateC,
				 LegalReference
				 ORDER BY 
				 UniqueID) Row_run
FROM Nashville
)
SELECT *
FROM RowNumCTE
WHERE Row_run > 1
ORDER BY PropertyAddress

--deleting the duplicates
WITH RowNumCTE AS(
SELECT *,
	ROW_NUMBER()OVER(
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDateC,
				 LegalReference
				 ORDER BY 
				 UniqueID) Row_run
FROM Nashville
)
DELETE
FROM RowNumCTE
WHERE Row_run > 1

--delete original owner address and property address
ALTER TABLE Nashville
DROP COLUMN PropertyAddress, OwnerAddress