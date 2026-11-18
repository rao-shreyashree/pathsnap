# Member B · Laptop 2 UI and KNN

## Run the UI

From the repository root, run:

~~~sh
python -m http.server 8080 --bind 127.0.0.1 --directory member_b/ui
~~~

Open http://127.0.0.1:8080. The UI uses a local coordinate plot and has no map-tile, CDN, or package downloads. Demo mode provides ten synthetic examples to check rendering offline.

For the two-laptop demonstration, connect both laptops to the same local network. Start the API on laptop 1, listening on its LAN interface. On laptop 2, switch to **Laptop 1 API**, enter the backend base URL such as http://192.168.1.10:8000, and click **Load trips**. If the backend provides only a prediction endpoint, enter the official evaluator trip IDs under **Enter trip IDs manually**. In FastAPI CORS configuration, allow the exact UI origin http://127.0.0.1:8080 and permit the GET and POST methods and Content-Type header.

## API handoff

- Optional GET /trips returns {"trips":[{"trip_id":"00000007","label":"Trip 07"}]}. Populate it from the server's authorized evaluator partition.
- POST /predict accepts {"trip_id":"00000007"}. Trip IDs are strings and preserve leading zeros.
- The response has trip_id, predicted_route (a GeoJSON LineString or MultiLineString in [longitude, latitude] order), and metrics (pointwise_accuracy as a fraction, correct_observations, scored_observations).
- Return aggregate metrics for the decoded route. Keep references and per-observation ground truth server-side. Return 404 for an unknown ID and use a JSON detail string for readable errors.

The browser has been tested with demo data and fixture API responses. Verify the live request against the teammate's FastAPI server and capture the live evaluation screenshots once the backend and evaluator trip IDs are available.

## KNN and the shared evaluation harness

Install dependencies with python -m pip install -r member_b/requirements.txt. Run:

~~~sh
python member_b/pilot_features.py "Student folder.zip" candidate_data
python member_b/evaluate.py candidate_data/train.csv candidate_data/student_test.csv member_b/results
~~~

The first command also accepts an extracted data directory containing train and student_test. The committed metrics use the prototype candidate and reference-alignment pipeline, four trip-grouped folds, all 80 training trips, and all 10 student-test trips. Observation top-1 was 78.12%, CV candidate macro F1 was 0.834, and student-test candidate macro F1 was 0.854. The chart and CSV are in results. See REVIEW_ONE.md for the baseline methodology and limitations.

For the final model comparison, use one feature table, candidate set, alignment, grouped-fraction policy and scoring method across all five models. Rerun the KNN baseline with that agreed setup before comparing its metrics to other team members' runs. For matching harness inputs and outputs see B_TO_C_HANDOFF.md.
