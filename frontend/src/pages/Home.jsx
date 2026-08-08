import { useEffect, useMemo, useState } from 'react'
import Header from '../components/Header.jsx'
import ImageUploader from '../components/ImageUploader.jsx'
import PredictionCard from '../components/PredictionCard.jsx'
import { getHealth, predictPlantDisease } from '../services/api.js'

function Home() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState(null)

  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ''), [file])

  useEffect(() => {
    getHealth().then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const handleFileSelect = (nextFile) => {
    setFile(nextFile)
    setResult(null)
    setError('')
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError('')
    try {
      setResult(await predictPlantDisease(file))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError('')
  }

  return (
    <div className="app-shell">
      <Header status={status} />
      <main className="app-main">
        <section className="hero">
          <div className="hero-content">
            <div className="eyebrow">CNN • EfficientNet-B3 • MSP + GOAD</div>
            <h1>PlantGuard AI</h1>
            <p>Upload a plant leaf image and get an AI-assisted disease prediction with confidence scores and treatment guidance.</p>
          </div>
          <div className="metric-row">
            <div className="metric"><strong>38</strong><span>PlantVillage classes</span></div>
            <div className="metric"><strong>70K+</strong><span>training images</span></div>
            <div className="metric"><strong>300px</strong><span>inference input</span></div>
            <div className="metric"><strong>{status?.anomaly_params_available ? 'Active' : 'Ready'}</strong><span>quality guard</span></div>
          </div>
        </section>

        <div className="workspace">
          <ImageUploader
            file={file}
            previewUrl={previewUrl}
            loading={loading}
            onFileSelect={handleFileSelect}
            onAnalyze={handleAnalyze}
            onReset={handleReset}
          />
          <PredictionCard result={result} />
        </div>

        {error && <div className="alert error">{error}</div>}
      </main>
    </div>
  )
}

export default Home
