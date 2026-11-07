import { ChangeDetectionStrategy, Component } from "@angular/core";

const RESOURCES = [
  {
    label: "Express docs",
    href: "https://a2ui.org/specification/proposals/express/",
  },
  { label: "Newsletter", href: "https://soverius.ai/newsletter" },
  { label: "Blog", href: "https://soverius.ai/blog" },
] as const;

/** Brand, title and resource links; the page projects its controls. */
@Component({
  selector: "app-page-header",
  changeDetection: ChangeDetectionStrategy.OnPush,
  host: { role: "banner" },
  template: `
    <div class="identity">
      <a
        class="brand"
        href="https://soverius.ai/"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Soverius AI home"
      >
        <img
          src="soverius-logo.svg"
          alt="Soverius AI"
          width="132"
          height="38"
        />
      </a>
      <h1>A2UI <span>1.0</span> Express</h1>
    </div>
    <nav class="header-links" aria-label="Demo resources">
      @for (link of resources; track link.href) {
        <a [href]="link.href" target="_blank" rel="noopener noreferrer"
          >{{ link.label }} ↗</a
        >
      }
    </nav>
    <ng-content />
  `,
  styleUrl: "./page-header.css",
})
export class PageHeader {
  protected readonly resources = RESOURCES;
}
