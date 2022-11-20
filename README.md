# GervigreindFinalProject

Data summary:
North America:
| Country | City | Number of listings | Exchange rate 09/22 (local cur to Euros)
| -------- | -------- | -------- | -------- |
| USA | Austin | 18,337 |  1.003 |
| USA | Boston | 5,185 |  1.003 |
| USA | Chicago | 7,414 |  1.003 |
| USA | Dallas | 6,546 |  1.003 |
| USA | Los Angeles | 41,815 |  1.003 |
| USA | New York City | 39,881 |  1.003 |
| USA | San Francisco | 6,629  |  1.003 |
| USA | Seattle | 5,904 |  1.003 |
| USA | Washington, D.C. | 6,473 |  1.003 |
| Canada | Montreal | 13,621 |  0.7501 |
| Canada | Toronto | 15,978 |  0.7501 |
| Canada | Vancouver | 5,572 |  0.7501 |

Europe:
| Country | City | Number of listings | Exchange rate 09/22 (local cur to Euros)
| -------- | -------- | -------- | -------- |
| Austria | Vienna | 11,797 | 1.00 |
| Belgium | Brussels | 6,065 | 1.00 |
| Czech Republic | Prague | 7,537 | 0.0406 |
| Denmark | Copenhagen | 13,815 | 0.1344 |
| France | Bordeax | 10,885 | 1.00 |
| France | Paris | 61,365 | 1.00 |
| France | Lyon | 10,934 | 1.00 |
| Germany | Berlin | 16,680 | 1.00 |
| Germany | Munich | 6,627 | 1.00 |
| Greece | Athens | 12,165 | 1.00 |
| Greece | Crete | 23,724 | 1.00 |
| Ireland | Dublin | 7,566 | 1.00 |
| Italy | Florence | 11,138 | 1.00 |
| Italy | Milan | 19,248 | 1.00 |
| Italy | Rome | 24,782 | 1.00 |
| Italy | Venice | 7,988 | 1.00 |
| Norway | Oslo | 5,371 | 0.097 |
| Portugal | Lisbon | 19,651 | 1.00 |
| Portugal | Porto | 11,804 | 1.00 |
| Spain | Barcelona | 16,920 | 1.00 |
| Spain | Madrid | 20,681 | 1.00 |
| Spain | Mallorca | 19,049 | 1.00 |
| Spain | Sevilla | 6,494 | 1.00 |
| Spain | Valencia | 7,355 | 1.00 |
| Sweden | Stockholm | 3,990 | 0.0921 |
| Switzerland | Geneva | 3,370 | 1.0403 |
| Switzerland | Zurich | 2,246 | 1.0403 |
| Netherlands | Amsterdam | 6,893 | 1.00 |
| Netherlands | Rotterdam | 1,042 | 1.00 |
| Turkey | Istanbul | 33,259 | 0.0547 |
| UK | Edinburgh | 7,818 | 1.1408 |
| UK | Greater Manchester | 4,341 | 1.1408 |
| UK | London | 69,351 | 1.1408  |

Meeting with Ana 8/11/2022:

Questions: Which data to select?

Use the free data. Start with one city and test it on same city and eval on another city. She believes that the features match up, she will check it for us.

Maybe thinking of moving towards SVR before NN?

YES thats a great strategy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

I would start with chosing the most valuable features. Size, bedrooms, location. Start without location, just area, no. of bedrooms, just the very simple features. 

Remember to convert currency correctly when evaluating.


Importante featurae!!


34 accommodates
35 bathrooms
37 bedrooms
38 beds


40 price


27 neighbourhood
28 neighbourhood_cleansed
29 neighbourhood_group_cleansed
30 latitude
31 longitude



32 property_type
33 room_type

36 bathrooms_text

39 amenities

41 minimum_nights
42 maximum_nights
43 minimum_minimum_nights
44 maximum_minimum_nights
45 minimum_maximum_nights
46 maximum_maximum_nights
47 minimum_nights_avg_ntm
48 maximum_nights_avg_ntm
49 calendar_updated
50 has_availability
51 availability_30
52 availability_60
53 availability_90
54 availability_365
55 calendar_last_scraped
56 number_of_reviews
57 number_of_reviews_ltm
58 number_of_reviews_l30d
59 first_review
60 last_review
61 review_scores_rating
62 review_scores_accuracy
63 review_scores_cleanliness
64 review_scores_checkin
65 review_scores_communication
66 review_scores_location
67 review_scores_value
68 license
69 instant_bookable
70 calculated_host_listings_count
71 calculated_host_listings_count_entire_homes
72 calculated_host_listings_count_private_rooms
73 calculated_host_listings_count_shared_rooms
74 reviews_per_month

0 id
1 listing_url
2 scrape_id
3 last_scraped
4 source
5 name
6 description
7 neighborhood_overview
8 picture_url
9 host_id
10 host_url
11 host_name
12 host_since
13 host_location
14 host_about
15 host_response_time
16 host_response_rate
17 host_acceptance_rate
18 host_is_superhost
19 host_thumbnail_url
20 host_picture_url
21 host_neighbourhood
22 host_listings_count
23 host_total_listings_count
24 host_verifications
25 host_has_profile_pic
26 host_identity_verified
27 neighbourhood
28 neighbourhood_cleansed
29 neighbourhood_group_cleansed
30 latitude
31 longitude
32 property_type
33 room_type
34 accommodates
35 bathrooms
36 bathrooms_text
37 bedrooms
38 beds
39 amenities
40 price
41 minimum_nights
42 maximum_nights
43 minimum_minimum_nights
44 maximum_minimum_nights
45 minimum_maximum_nights
46 maximum_maximum_nights
47 minimum_nights_avg_ntm
48 maximum_nights_avg_ntm
49 calendar_updated
50 has_availability
51 availability_30
52 availability_60
53 availability_90
54 availability_365
55 calendar_last_scraped
56 number_of_reviews
57 number_of_reviews_ltm
58 number_of_reviews_l30d
59 first_review
60 last_review
61 review_scores_rating
62 review_scores_accuracy
63 review_scores_cleanliness
64 review_scores_checkin
65 review_scores_communication
66 review_scores_location
67 review_scores_value
68 license
69 instant_bookable
70 calculated_host_listings_count
71 calculated_host_listings_count_entire_homes
72 calculated_host_listings_count_private_rooms
73 calculated_host_listings_count_shared_rooms
74 reviews_per_month



