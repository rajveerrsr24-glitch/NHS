from flask import Flask, request, render_template, abort, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import io
import base64

app = Flask(__name__)

# Global model stats for chatbot
model_stats = {
    'slope': None,
    'intercept': None,
    'r2': None
}

def predict_for_dates(model, dates):
    ordinals = [[d.toordinal()] for d in dates]
    predictions = model.predict(ordinals)
    return {d.strftime('%Y-%m'): round(p) for d, p in zip(dates, predictions)}

@app.route('/', methods=['GET'])
def index():
    return render_template('input.html')

@app.route('/linear_regression', methods=['POST'])
def linear_regression():
    if 'file' not in request.files:
        abort(403)
    file = request.files['file']

    try:
        # Read Excel file and detect sheet names
        xls = pd.ExcelFile(file)
        sheet_name = xls.sheet_names[0]  # Use the first sheet
    except Exception:
        abort(403)

    try:
        # Try multi-header format
        df = pd.read_excel(xls, sheet_name=sheet_name, header=[12, 13])
        df = df.drop(columns=df.columns[0])
        df.columns = [' '.join([str(x) for x in col if pd.notna(x)]).strip() for col in df.columns]
    except Exception:
        # Fallback to single-header flat format
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=12)
        except Exception:
            abort(403)

    # Try to find matching columns
    week_col = next((c for c in df.columns if 'Week Ending' in c), None)
    wait_col = next((c for c in df.columns if 'Total Waiting List' in c), None)
    if not week_col or not wait_col:
        abort(403)

    try:
        df = df.dropna(subset=[week_col, wait_col])
        df[week_col] = pd.to_datetime(df[week_col], errors='coerce')
        df = df.dropna(subset=[week_col])
        df = df.sort_values(by=week_col)

        y = df[wait_col].astype(float).values
        X = df[week_col].map(pd.Timestamp.toordinal).values.reshape(-1, 1)

        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        slope = model.coef_[0]
        intercept = model.intercept_
        r2 = r2_score(y, y_pred)

        model_stats['slope'] = slope
        model_stats['intercept'] = intercept
        model_stats['r2'] = r2

        # Plotting
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        ax.plot(df[week_col], y, 'o', color='#64ffda', label='Actual', markersize=5)
        ax.plot(df[week_col], y_pred, '-', color='#1f6feb', linewidth=2.5, label='Prediction')

        ax.set_title('NHS Waiting List Forecast', fontsize=18, weight='bold', color='white', pad=20)
        ax.set_xlabel('Date', fontsize=12, labelpad=15)
        ax.set_ylabel('Number of Patients', fontsize=12, labelpad=15)
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')

        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        for spine in ax.spines.values():
            spine.set_color('white')

        ax.text(0.02, 0.95, f"y = {slope:.2f}x + {intercept:.2f}", transform=ax.transAxes, fontsize=12, color='#bbbbbb')
        ax.text(0.02, 0.89, f"$R^2 = {r2:.4f}$", transform=ax.transAxes, fontsize=12, color='#bbbbbb')
        ax.legend(loc='upper left', fontsize=12)

        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        plot_data = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        future_dates = pd.date_range(start='2026-01-01', end='2030-12-01', freq='MS')
        future_predictions = predict_for_dates(model, future_dates)

        return render_template(
            'output.html',
            plot_url=f'data:image/png;base64,{plot_data}',
            predictions=future_predictions
        )

    except Exception:
        abort(403)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "").lower()

    slope = model_stats.get('slope')
    intercept = model_stats.get('intercept')
    r2 = model_stats.get('r2')

    if not all([slope, intercept, r2]):
        return jsonify({"answer": "The model hasn't been run yet. Please upload data first."})

    if "growth rate" in message or "slope" in message:
        answer = f"📈 The current growth rate (slope) is approximately {slope:.2f} patients per day."
    elif "r value" in message or "r²" in message or "r2" in message:
        answer = f"📊 The model's R² value is {r2:.4f}, which indicates how well the model fits the data."
    elif "summary" in message:
        answer = (
            f"📄 The model uses linear regression on historical NHS waiting list data.\n"
            f"Equation: y = {slope:.2f}x + {intercept:.2f}\n"
            f"R²: {r2:.4f}"
        )
    else:
        answer = "🤖 Sorry, I didn't understand that. Try asking about 'growth rate', 'R value', or 'summary'."

    return jsonify({"answer": answer})

@app.errorhandler(403)
def forbidden(e):
    return "403 Forbidden — please check your upload and try again.", 403

if __name__ == '__main__':
    app.run(debug=True)
