import { Activity, AlertTriangle } from 'lucide-react'
import ConfidenceBar from './ConfidenceBar.jsx'

function formatClassName(value) {
  return value?.replaceAll('___', ' - ').replaceAll('_', ' ') || 'Unknown'
}

function PredictionCard({ result }) {
  if (!result) {
    return (
      <section className="panel">
        <div className="empty-result">
          <div>
            <Activity size={38} />
            <h2>Prediction results will appear here</h2>
            <p>Upload a leaf image and run the model to view confidence, top classes, and treatment guidance.</p>
          </div>
        </div>
      </section>
    )
  }

  const confidence = result.confidence || result.msp_confidence || 0
  const isHealthy = result.severity === 'none'
  const badgeClass = result.is_anomaly ? 'unknown' : isHealthy ? 'healthy' : 'disease'
  const badgeLabel = result.is_anomaly ? 'Unknown image' : isHealthy ? 'Healthy' : 'Disease detected'
  const title = result.is_anomaly ? 'Unknown image pattern' : result.condition

  return (
    <section className="panel">
      <div className="result-card">
        <div className="result-top">
          <div>
            <div className="label-kicker">{result.plant || 'Quality guard'}</div>
            <h2 className="result-title">{title}</h2>
          </div>
          <span className={`badge ${badgeClass}`}>{badgeLabel}</span>
        </div>

        <p className="result-copy">
          {result.is_anomaly ? result.reason : result.description}
        </p>

        <div className="confidence-block">
          <div className="confidence-row">
            <div className="confidence-value">{(confidence * 100).toFixed(1)}%</div>
            <div className="confidence-label">confidence</div>
          </div>
          <ConfidenceBar value={confidence} />
        </div>

        {result.is_anomaly ? (
          <div className="alert error">
            <AlertTriangle size={18} /> Use a sharper close-up image with one leaf in frame.
          </div>
        ) : (
          <>
            <div className="top-list">
              {result.top5?.map((item, index) => (
                <div className="top-row" key={item.class || item.class_name}>
                  <span>{index + 1}</span>
                  <strong>{formatClassName(item.class || item.class_name)}</strong>
                  <em>{(item.probability * 100).toFixed(1)}%</em>
                  <ConfidenceBar value={item.probability} compact />
                </div>
              ))}
            </div>
            <div className="recommend">
              <h3>Treatment guidance</h3>
              <p>{result.treatment}</p>
            </div>
          </>
        )}
      </div>
    </section>
  )
}

export default PredictionCard
