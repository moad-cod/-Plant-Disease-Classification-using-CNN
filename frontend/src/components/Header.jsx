import { Leaf } from 'lucide-react'

function Header({ status }) {
  const label = status?.model_available ? 'Model ready' : 'API connected'

  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand">
          <span className="brand-mark">
            <Leaf size={22} strokeWidth={2.4} />
          </span>
          <span>PlantGuard AI</span>
        </div>
        <div className="header-status">
          <span className="status-dot" />
          {label}
        </div>
      </div>
    </header>
  )
}

export default Header
