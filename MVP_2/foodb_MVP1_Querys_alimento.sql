-- ** FOOD QUERYS ** --


-- Pepper (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 40 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
AND cont.orig_food_id = 02009
ORDER BY source_id;


-- Coffee (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 58 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Turmeric (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 68 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Mentha (FAIL)
SELECT * -- cont.source_id, cont.orig_source_name, 
-- cont.food_id, cont.orig_food_common_name, cont.orig_content, 
-- cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 109 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Rosemary (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 159 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Common sage (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 165 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Winter savory (NO FULL)
SELECT * -- cont.source_id, cont.orig_source_name, 
-- cont.food_id, cont.orig_food_common_name, cont.orig_content, 
-- cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 168 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Common thyme (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 183 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Common wheat (NO FULL)
SELECT * -- cont.source_id, cont.orig_source_name, 
-- cont.food_id, cont.orig_food_common_name, cont.orig_content, 
-- cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 187 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.orig_food_common_name = 'Wheat'
GROUP BY source_id;


-- Ginger (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 206 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Ginseng (FAIL)
SELECT * -- cont.source_id, cont.orig_source_name, 
	-- cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	-- cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 219 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Ginkgo nuts (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 372 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE '%dried%'
ORDER BY source_id;


-- Horseradish tree (NO FULL)
SELECT * -- cont.source_id, cont.orig_source_name, 
	-- cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	-- cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 384 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.orig_food_common_name LIKE  '%Drumstick leaves, raw%'
ORDER BY source_id;


-- Shiitake (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 562 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.preparation_type LIKE  '%dried%'
ORDER BY source_id;


-- Maitake (OK)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 569 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.preparation_type LIKE  '%dried%'
ORDER BY source_id;


-- Cinnamon (NO FULL)
SELECT * -- cont.source_id, cont.orig_source_name, 
	-- cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	-- cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 586 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.preparation_type LIKE  '%dried%'
GROUP BY source_id 
ORDER BY source_id;


-- Salt (FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content 
FROM foodb.contents cont
WHERE cont.food_id = 666 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.orig_food_common_name LIKE  '%table%'
AND orig_food_id = 02047
ORDER BY source_id;


-- Chocolate (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content), 
	AVG(cont.orig_min), AVG(cont.orig_max), cont.orig_unit, 
	AVG(cont.standard_content)
FROM foodb.contents cont
WHERE cont.food_id = 709 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.orig_food_common_name LIKE '%milk'
GROUP BY source_id 
ORDER BY source_id;


-- Hibiscus tea (FULL)
SELECT */*cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, cont.orig_content, 
	cont.orig_min, cont.orig_max, cont.orig_unit, cont.standard_content */
FROM foodb.contents cont
WHERE cont.food_id = 748 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.orig_food_common_name LIKE '%milk'
-- GROUP BY source_id 
ORDER BY source_id;


-- Green Tea (NO FULL)
SELECT cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content), 
	AVG(cont.orig_min), AVG(cont.orig_max), cont.orig_unit, 
	AVG(cont.standard_content)
FROM foodb.contents cont
WHERE cont.food_id = 940 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
AND cont.orig_food_common_name LIKE '%leaves%'
GROUP BY cont.source_id 
ORDER BY cont.source_id;


-- Guarana (FAIL)
SELECT * /*cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content), 
	AVG(cont.orig_min), AVG(cont.orig_max), cont.orig_unit, 
	AVG(cont.standard_content)*/
FROM foodb.contents cont
WHERE cont.food_id = 947 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.orig_food_common_name LIKE '%leaves%'
-- GROUP BY cont.source_id 
ORDER BY cont.source_id;


-- Mate (FAIL)
SELECT * /*cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content), 
	AVG(cont.orig_min), AVG(cont.orig_max), cont.orig_unit, 
	AVG(cont.standard_content)*/
FROM foodb.contents cont
WHERE cont.food_id = 948 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.orig_food_common_name LIKE '%leaves%'
-- GROUP BY cont.source_id 
ORDER BY cont.source_id;


-- Coconut milk (FAIL)
SELECT * /*cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content), 
	AVG(cont.orig_min), AVG(cont.orig_max), cont.orig_unit, 
	AVG(cont.standard_content)*/
FROM foodb.contents cont
WHERE cont.food_id = 970 AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.orig_food_common_name LIKE '%leaves%'
-- GROUP BY cont.source_id 
ORDER BY cont.source_id;


-- Coconut oil (FAIL)
SELECT * /*cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content), 
	AVG(cont.orig_min), AVG(cont.orig_max), cont.orig_unit, 
	AVG(cont.standard_content)*/
FROM foodb.contents cont
WHERE cont.food_id = 973/* AND cont.source_type = 'Compound' 
AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)
-- AND cont.orig_food_common_name LIKE '%leaves%'
-- GROUP BY cont.source_id 
ORDER BY cont.source_id*/;


-- Goji (FAIL)
SELECT * /*cont.source_id, cont.orig_source_name, 
	cont.food_id, cont.orig_food_common_name, AVG(cont.orig_content), 
	AVG(cont.orig_min), AVG(cont.orig_max), cont.orig_unit, 
	AVG(cont.standard_content)*/
FROM foodb.contents cont
WHERE cont.food_id = 1014 AND cont.source_type = 'Compound' 
/*AND cont.source_id IN (574, 710, 1014, 1223, 2100, 3514, 3519, 3521, 
					   3522, 3524, 3583, 3716, 3730, 8425, 12163, 
					   13393, 13403, 14507, 16258, 23049, 23250)*/
-- AND cont.orig_food_common_name LIKE '%leaves%'
-- GROUP BY cont.source_id 
ORDER BY cont.source_id;


